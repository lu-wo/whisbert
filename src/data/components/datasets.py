import random
import os, sys

import torch
from torch.utils.data import Dataset
import numpy as np
from collections.abc import Iterable
from typing import List, Tuple, Union
from tqdm import tqdm
from typing import List
import re
import soundfile as sf
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import time
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

from transformers import Wav2Vec2FeatureExtractor

from src.utils.audio_utils import get_flac_files, read_flac_file
from src.utils.audio_utils import MyASTFeatureExtractor
from src.utils.text_processing import read_peoples_speech_txtfile

# default separators for the tokenization
SEP = ["-", ".", ",", ";"]


def assign_labels(input_string, labels):
    # Create list to hold words and punctuation
    words_with_punctuation = re.findall(r"[\w']+|[.,!?;\"-]|'", input_string)

    # Create list to hold only words
    words_only = re.findall(r"\w+'?\w*", input_string)

    # Make sure the number of labels matches the number of words
    if not len(labels) == len(words_only):
        # print(
        #     f"Aligning labels: Number of labels ({len(labels)}) does not match number of words ({len(words_only)})"
        # )
        # alignmend or extraction failed, skip sample
        return None, None, None

    # Create a generator for word-label pairs
    word_label_pairs = ((word, label) for word, label in zip(words_only, labels))

    # Create list of tuples where each word is matched to a label and each punctuation is matched to None
    words_with_labels = []
    for token in words_with_punctuation:
        if re.match(r"\w+'?\w*", token):
            words_with_labels.append(next(word_label_pairs))
        else:
            words_with_labels.append((token, None))

    return words_only, words_with_punctuation, words_with_labels


def tokenize_text_with_labels(
    text: List[str],
    # aligned_words: List[str],
    labels: Union[List[float], List[List[float]]],
    tokenizer,
    model_type,
    invalid_label: int = -999,
    score_first_token: bool = False,
    score_last_token: bool = False,
    relative_to_prev: bool = False,
    n_prev: int = 1,
    relative_to_mean=False,
    word_stats: dict = None,
    add_prefix_space: bool = True,
    min_words=4,
    max_words=60,
):
    """
    Tokenize the input text and associate labels with each token.

    Args:
        text (str): The input text to tokenize.
        labels (list): A list of labels corresponding to each word in the text.
        model_type (str): The type of the language model (e.g., 'gpt2', 'bert-uncased', 'bert-cased').
        invalid_label (int, optional): The label to assign to invalid tokens (e.g., punctuation and whitespace). Defaults to -1.
        score_first_token (bool, optional): If True, only the first token of a multi-token word will have a mask value of 1. Defaults to True.
        relative_to_prev (bool, optional): If True, adjust the labels to be relative to the average of the previous n_prev words. Defaults to False.
        n_prev (int, optional): The number of previous words to consider when adjusting labels. Only used if relative_to_prev is True. Defaults to 1.
        relative_to_mean (bool, optional): If True, adjust the labels to be relative to the mean of the word in the corpus passed by the dict. Defaults to False.
        word_stats (dict, optional): A dictionary containing word statistics and as such the mean prominence of each word in the corpus. Defaults to None.

    Returns:
        tuple: A tuple containing the following elements:
            - input_text (str): The input text.
            - tokenized_text (list): The tokenized text as a list of tokens.
            - tokenized_labels (list): The list of labels corresponding to each token in the tokenized_text.
            - token_ids (list): The list of token IDs corresponding to each token in the tokenized_text.
            - mask (list): A binary mask indicating whether a token should be scored (1) or not (0).
    """
    assert not (
        score_first_token and score_last_token
    ), "Only one of score_first_token and score_last_token can be True"

    # check if we have vector-valued labels and if so, adapt the invalid label to it
    if isinstance(labels[0], Iterable) and not isinstance(labels[0], str):
        invalid_label = [invalid_label] * len(labels[0])

    # remove None labels
    labels = [l for l in labels if l is not None]
    # print(f"Text before : {text}")
    # print(f"Labels: {labels}")
    # print(f"Length labels: {len(labels)}")

    _, _, labeled_tokens = assign_labels(text, labels)
    # print(f"Labeled tokens: {labeled_tokens}")
    # check if label assignment is possible
    if labeled_tokens is None:
        return None

    # apply cleaning on the number of words to remove outliers
    if len(labeled_tokens) < min_words or len(labeled_tokens) > max_words:
        # print(f"Text: {text}")
        return None

    word_units, labels = zip(*labeled_tokens)
    words = list(word_units)
    labels = list(labels)
    original_labels = labels  # store them to return them later

    # from here on we assume that each word is a unit that has a (potentially None) label
    assert len(words) == len(labels), "The number of words and labels should be equal"

    # if relative_to_prev is True, we adjust the labels to be relative to the average of the previous n_prev words
    if relative_to_prev and not relative_to_mean:
        new_labels = []
        # print(f"labels before: {labels}")
        for i, label in enumerate(labels):
            if i < n_prev or label is None:
                new_labels.append(label)
            else:
                # Get the previous n_prev labels which are not None
                prev_labels = [
                    labels[j] for j in range(i - n_prev, i) if labels[j] is not None
                ]
                # print(f"i = {i}, label {label}, prev_labels = {prev_labels}")
                if prev_labels:
                    avg_prev = sum(prev_labels) / len(prev_labels)
                    new_labels.append(label - avg_prev)
                else:
                    new_labels.append(label)
        labels = new_labels
        # print(f"labels after: {labels}")

    # if relative_to_mean is True, we adjust the labels to be relative to the mean of the word in the corpus
    elif relative_to_mean:
        if word_stats is None:
            raise ValueError(
                "Word statistics are required for relative_to_mean method."
            )
        new_labels = []
        for word, label in zip(words, labels):
            if label is None:
                new_labels.append(label)
                continue

            if word in word_stats:
                mean_label = word_stats[word]["mean"]
            elif word.lower() in word_stats:
                mean_label = word_stats[word.lower()]["mean"]
            else:
                mean_label = word_stats["$global$"]["mean"]
            new_labels.append(label - mean_label)
        labels = new_labels

    tokenized_text, tokenized_labels, token_ids, mask, word_to_tokens = (
        [],
        [],
        [],
        [],
        [],
    )

    # if model is Bert we add a [CLS] token at the beginning
    if model_type.lower().startswith("bert"):
        tokenized_text.append(tokenizer.cls_token)
        tokenized_labels.append(invalid_label)
        token_ids.append(tokenizer.cls_token_id)
        mask.append(0)

    # we tokenize each word separately and keep track of the mapping between words and tokens
    for i, (word, label) in enumerate(zip(words, labels)):
        # TODO: remove this hardcoded hack for gpt, must be similar for llama
        if (
            "gpt" in model_type.lower()
            and i > 0
            and not np.any([s in word for s in SEP])  # check for punctuation
        ):
            word = " " + word
        # else:
        # print("word: ", word)
        # print(f"model name {model_type}")
        # print(f"i: {i}")
        # print(f"SEP: {SEP}")

        tokens = tokenizer.tokenize(word)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        tokenized_text.extend(tokens)
        token_ids.extend(ids)
        word_to_tokens.extend((word, ids))

        if score_first_token:
            mask.extend([1] + [0] * (len(tokens) - 1))
            tokenized_labels.extend([label] + [invalid_label] * (len(tokens) - 1))
        elif score_last_token:
            mask.extend([0] * (len(tokens) - 1) + [1])
            tokenized_labels.extend([invalid_label] * (len(tokens) - 1) + [label])
        else:
            mask.extend([1] * len(tokens))
            tokenized_labels.extend([label] * len(tokens))

    # if model is BERT we add a [SEP] token at the end
    if model_type.lower().startswith("bert"):
        tokenized_text.append(tokenizer.sep_token)
        tokenized_labels.append(invalid_label)
        token_ids.append(tokenizer.sep_token_id)
        mask.append(0)

    # substitute all None in labels (could not compute a label here) with invalid_label as well as set mask to 0 at these positions
    tokenized_labels = [
        label if label is not None else invalid_label for label in tokenized_labels
    ]
    # mask = [1 if label != invalid_label else 0 for label in tokenized_labels]
    mask = [1 if np.all(label != invalid_label) else 0 for label in tokenized_labels]

    # if mask is all 0 (no valid predicitons) we return None
    if np.all(mask == 0):
        return None

    return (
        text,
        tokenized_text,
        original_labels,
        tokenized_labels,
        token_ids,
        mask,
        word_to_tokens,
    )


class TokenTaggingDataset(Dataset):
    def __init__(
        self,
        input_texts,
        targets,
        tokenizer,
        model_name: str,
        score_first_token: bool = False,
        score_last_token: bool = False,
        relative_to_prev: bool = False,
        n_prev: int = 1,
        relative_to_mean=False,
        word_stats: dict = None,
        debug: bool = False,
    ):
        """
        ::param inputs: list of strings
        ::param targets: list of lists of labels
        ::param model_name: name of the model to use
        ::param tokenizer: tokenizer object
        ::param score_first_token: whether to score only the first token of a word
        ::param relative_to_prev: whether to score relative to the previous token
        ::param n_prev: number of previous tokens to consider
        """
        self.inputs = input_texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.score_first_token = score_first_token
        self.score_last_token = score_last_token
        self.relative_to_prev = relative_to_prev
        self.n_prev = n_prev
        self.relative_to_mean = relative_to_mean
        self.word_stats = word_stats
        self.debug = debug

        cnt_failed = 0
        # Perform preprocessing at initialization
        self.processed_data = []
        for text, labels_per_word in tqdm(
            zip(self.inputs, self.targets),
            total=len(self.inputs),
            desc="Preprocessing samples",
        ):
            result = tokenize_text_with_labels(
                text=text,
                labels=labels_per_word,
                tokenizer=self.tokenizer,
                model_type=self.model_name,
                score_first_token=self.score_first_token,
                score_last_token=self.score_last_token,
                relative_to_prev=self.relative_to_prev,
                n_prev=self.n_prev,
                relative_to_mean=self.relative_to_mean,
                word_stats=self.word_stats,
            )

            if not result:
                cnt_failed += 1
                continue

            (
                input_text,
                tokenized_text,
                original_labels,
                tokenized_labels,
                token_ids,
                mask,
                word_to_tokens,
            ) = result

            self.processed_data.append(
                {
                    "input_text": input_text,
                    "tokenized_text": tokenized_text,
                    "original_labels": original_labels,
                    "tokenized_labels": tokenized_labels,
                    "input_ids": token_ids,
                    "loss_mask": mask,
                    "word_to_tokens": word_to_tokens,
                }
            )

        print(f"Failed {cnt_failed}/{len(self.inputs)}")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]

        if self.debug:
            print("---")
            print("input_text", item["input_text"])
            print("tokenized_text", item["tokenized_text"])
            print("original_labels", item["original_labels"])
            print("tokenized_labels", item["tokenized_labels"])
            print("input_ids", item["input_ids"])
            print("loss_mask", item["loss_mask"])
            print("word_to_tokens", item["word_to_tokens"])

        return item


class StringDataset(Dataset):
    def __init__(self, list_of_strings):
        self.data = list_of_strings
        print(f"Constructed string dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item


class TokenizedStringDataset(Dataset):
    def __init__(self, list_of_strings, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.data = [
            self.tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_attention_mask=True,
            )
            for text in list_of_strings
        ]
        print(f"Constructed tokenized string dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["input_ids"]


class TimeSeriesDataset(Dataset):
    def __init__(self, data, texts=None):
        self.data = data
        self.texts = texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = torch.tensor(self.data[index], dtype=torch.float32)
        if self.texts is not None:
            text = self.texts[index]
            return sequence, text
        return sequence


class AudioDataset(Dataset):
    def __init__(
        self,
        flac_files: List[str] = None,
        root_dir: str = None,
        sr: int = 16000,
        min_seq_len: int = 10,
        max_seq_len: int = 10,
    ):
        """Initialize the AudioDataset.

        Args:
            root_dir (str): The root directory containing the flac files.
            sr (int, optional): The desired sampling rate. Defaults to 16000.
            sequence_length (int, optional): The maximum sequence length in seconds. Defaults to 10.
        """
        self.root_dir = root_dir
        self.sr = sr
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

        self.expected_sequence_length = (
            self.min_seq_len + self.max_seq_len
        ) / 2  # in seconds
        self.flac_files = flac_files if flac_files else get_flac_files(root_dir)
        self.total_samples = (
            len(self.flac_files) * 4e7
        )  # TODO: remove this hardcode self.calculate_total_samples()
        print(f"Dataset has {self.__len__()} samples")

        self.current_data = self.load_random_file()
        self.current_offset = 0

    def calculate_total_samples(self):
        """Calculate the total number of samples in all FLAC files.

        Returns:
            int: The total number of samples.
        """
        total_samples = 0
        print(f"Counting samples...")
        start_time = time.time()
        for file_path in tqdm(
            self.flac_files, total=len(self.flac_files), desc="Counting samples"
        ):
            data = read_flac_file(flac_file_path=file_path, sr=self.sr)["signal"]
            total_samples += len(data)
            print(f"length of data: {len(data)}")
        print(f"Counting took {time.time() - start_time} seconds")
        return total_samples

    def load_random_file(self):
        """Load a random flac file and resample it.

        Returns:
            np.ndarray: The resampled audio data.
        """
        # Select a random file
        file_path = np.random.choice(self.flac_files)
        return read_flac_file(flac_file_path=file_path, sr=self.sr)["signal"]

    def __getitem__(self, index, verbose=False):
        """Get a random audio sequence from the dataset.

        Args:
            index (int): The index (not used).

        Returns:
            torch.Tensor: The audio sequence tensor.
        """
        # Compute random sequence length in seconds
        random_length = np.random.uniform(self.min_seq_len, self.max_seq_len)
        num_samples = int(self.sr * random_length)

        # If remaining samples in current file is less than num_samples, load a new file
        if len(self.current_data) - self.current_offset < num_samples:
            self.current_data = self.load_random_file()
            self.current_offset = 0  # Reset the offset when loading a new file

        # Get the audio sequence
        sequence = self.current_data[
            self.current_offset : self.current_offset + num_samples
        ]

        # Increase the offset by the number of samples used
        self.current_offset += num_samples

        # return numpy array
        return sequence

    def __len__(self):
        """Get the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return int(self.total_samples // int(self.sr * self.expected_sequence_length))


class PeoplesMultiModalDataset(Dataset):
    def __init__(
        self,
        alignment_files: List[str] = None,
        alignment_root_dir: str = None,
        flac_files: List[str] = None,
        audio_root_dir: str = None,
        sr: int = 16000,
        num_words_per_sample: int = 90,
        dataset_total_words: int = 88e6,
        unmatched_samples: bool = False,
    ):
        assert len(alignment_files) == len(flac_files)
        print(f"Loading {len(alignment_files)} alignment files")
        self.sr = sr
        self.num_words_per_sample = num_words_per_sample
        self.dataset_total_words = dataset_total_words
        self.unmatched_samples = unmatched_samples

        self.alignment_files = alignment_files
        self.flac_files = flac_files

        self.curr_file_index = 0
        self.curr_audio_file, self.curr_txt_file = self._load_next_files()
        self.curr_audio_offset, self.curr_txt_offset = 0, 0

        print(f"Dataset has {self.__len__()} samples")

    def _load_next_files(self):
        # start_t = time.time()

        print(f"Loading files {self.curr_file_index}")
        flac_file_path = self.flac_files[self.curr_file_index]
        flac_file = read_flac_file(flac_file_path=flac_file_path, sr=self.sr)["signal"]

        txt_file_path = self.alignment_files[self.curr_file_index]
        txt_file = read_peoples_speech_txtfile(txt_file_path)

        # print(f"Loading files took {time.time() - start_t} seconds")

        self.curr_file_index = (self.curr_file_index + 1) % len(
            self.flac_files
        )  # cycle through files

        print(f"File lengths: {len(flac_file)}, {len(txt_file)}")

        return flac_file, txt_file

    # rest of your class

    def _load_random_files(self):
        # start_t = time.time()
        random_index = np.random.randint(0, len(self.flac_files))

        flac_file_path = self.flac_files[random_index]
        flac_file = read_flac_file(flac_file_path=flac_file_path, sr=self.sr)["signal"]

        txt_file_path = self.alignment_files[random_index]
        txt_file = read_peoples_speech_txtfile(txt_file_path)

        # print(f"Loading files took {time.time() - start_t} seconds")

        print(f"File lengths: {len(flac_file)}, {len(txt_file)}")

        return flac_file, txt_file

    def _get_text_from_dict(self, cnt: List[dict], start_idx: int, end_idx: int):
        return " ".join([c["word"] for c in cnt[start_idx:end_idx]])

    def __getitem__(self, index):
        # print(f"Getting item {index}")
        # print(f"Current text offset: {self.curr_txt_offset}")
        # print(f"Current text file length: {len(self.curr_txt_file)}")

        # Check if there are enough words left in the current text file
        if (
            self.curr_txt_offset + self.num_words_per_sample
            > len(self.curr_txt_file) - 1
        ):
            # If not, load a new audio and text file
            # self.curr_audio_file, self.curr_txt_file = self._load_random_files()
            self.curr_audio_file, self.curr_txt_file = self._load_next_files()
            self.curr_audio_offset, self.curr_txt_offset = 0, 0
            # print(f"New file \nCurrent text offset: {self.curr_txt_offset}")
            # print(f"Current text file length: {len(self.curr_txt_file)}")

        # # Sample the words from the text file
        # sampled_words = self.curr_txt_file[
        #     self.curr_txt_offset : self.curr_txt_offset + self.num_words_per_sample
        # ]
        text = self._get_text_from_dict(
            self.curr_txt_file,
            self.curr_txt_offset,
            self.curr_txt_offset + self.num_words_per_sample,
        )

        # print(f"text: {text}")

        # Get the start and end times of the string in seconds
        start_time = self.curr_txt_file[self.curr_txt_offset]["start"]
        end_time = self.curr_txt_file[self.curr_txt_offset + self.num_words_per_sample][
            "end"
        ]
        # print(f"nb words: {self.num_words_per_sample}")
        if self.unmatched_samples:
            # print(f"Unmatched samples")
            length = int((end_time - start_time) * self.sr)
            if length > len(self.curr_audio_file):
                # need new file
                self.curr_audio_file, self.curr_txt_file = self._load_next_files()
                self.curr_audio_offset, self.curr_txt_offset = 0, 0
                return self.__getitem__(index)
            start_index = np.random.randint(0, len(self.curr_audio_file) - length)
            end_index = int(start_index + length)
        else:
            # Convert the start and end times to sample indices
            start_index = int(start_time * self.sr)
            end_index = int(end_time * self.sr)
            length = end_index - start_index

        # print(f"Audio file length: {length}")
        # print(f"Start index {start_index}, end index {end_index}")

        # Sample the corresponding audio segment
        audio_segment = self.curr_audio_file[start_index:end_index]
        # Update the audio offset
        self.curr_audio_offset = end_index
        # Update the text offset
        self.curr_txt_offset += self.num_words_per_sample

        return audio_segment, text

    def __len__(self):
        """Get the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return int(self.dataset_total_words // self.num_words_per_sample)


class PeoplesMultiModalPackagedDataset:
    def __init__(
        self,
        dataset: PeoplesMultiModalDataset,
        save_dir: str,
        num_files_per_pack: int = 10000,
    ):
        self.dataset = dataset
        self.num_files_per_pack = num_files_per_pack
        self.save_dir = save_dir
        self.current_pack_index = 0
        self.current_pack = None

        # Create save directory if it does not exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Check if there's a pack in the directory, if not, prepare packs
        if not self.is_pack_in_directory():
            print("No pack found in directory. Preparing packs now.")
            self.prepare_packs()

        self.load_pack()  # Load the first pack upon initialization

    def is_pack_in_directory(self):
        """Check if there's a pack file in the save directory."""
        files_in_directory = os.listdir(self.save_dir)
        pack_files = [
            file
            for file in files_in_directory
            if file.startswith("sample_pack") and file.endswith(".pkl")
        ]
        return len(pack_files) > 0

    def prepare_packs_seq(self):
        total_packs = len(self.dataset) // self.num_files_per_pack
        for i in tqdm(range(total_packs), desc="Preparing packs"):
            pack = [
                self.dataset[j]
                for j in range(
                    i * self.num_files_per_pack, (i + 1) * self.num_files_per_pack
                )
            ]
            print(f"Storing pack {i}")
            with open(os.path.join(self.save_dir, f"sample_pack_{i}.pkl"), "wb") as f:
                pickle.dump(pack, f)
            print(f"Finished storing pack {i}")

    def _prepare_pack(self, i):
        print(f"Start preparing pack {i}")
        pack = [
            self.dataset[j]
            for j in range(
                i * self.num_files_per_pack, (i + 1) * self.num_files_per_pack
            )
        ]
        print(f"Storing pack {i}")
        with open(os.path.join(self.save_dir, f"sample_pack_{i}.pkl"), "wb") as f:
            pickle.dump(pack, f)
        print(f"Finished storing pack {i}")

    def prepare_packs(self):
        total_packs = len(self.dataset) // self.num_files_per_pack

        # Prepare for multiprocessing
        cores = mp.cpu_count()
        pool = Pool(cores if cores <= 8 else 8)  # use up to 8 cores

        print("Starting parallel pack preparation.")
        for _ in tqdm(
            pool.imap_unordered(self._prepare_pack, range(total_packs)),
            total=total_packs,
        ):
            pass

        pool.close()
        pool.join()

    def load_pack(self):
        pack_path = os.path.join(
            self.save_dir, f"sample_pack_{self.current_pack_index}.pkl"
        )
        if os.path.exists(pack_path):
            with open(pack_path, "rb") as f:
                self.current_pack = pickle.load(f)
                self.current_pack_index += 1
        else:
            raise Exception("Pack file does not exist. Please prepare the packs first.")

    def __getitem__(self, index):
        if index >= len(self.current_pack):
            self.load_pack()
            index = 0  # Reset index for the new pack
        return self.current_pack[index]

    def __len__(self):
        """Get the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.dataset)


class PSPackagedDataset(Dataset):
    def __init__(
        self,
        pack_filepaths: list,
        num_files_per_pack: int = 10000,
        unmatch_samples: bool = False,
    ):
        self.pack_filepaths = pack_filepaths
        self.num_files_per_pack = num_files_per_pack
        self.unmatch_samples = unmatch_samples
        self.current_pack_index = 0
        self.current_pack = None
        self.cnt = 0

        print(
            f"Created pack dataset with {len(self.pack_filepaths)} packs and {len(pack_filepaths) * num_files_per_pack} samples."
        )

        self.load_pack()  # Load the first pack upon initialization

    def is_pack_in_directory(self):
        """Check if there's a pack file in the save directory."""
        pack_files = [
            file
            for file in self.pack_filepaths
            if file.startswith("sample_pack") and file.endswith(".pkl")
        ]
        return len(pack_files) > 0

    def load_pack(self):
        # TODO: get rid of this loop, it just skips corrupt packs
        failed_cnt = 0
        while self.current_pack_index < len(self.pack_filepaths):
            print(f"Loading pack {self.current_pack_index}")
            pack_path = self.pack_filepaths[self.current_pack_index]
            try:
                if os.path.exists(pack_path):
                    with open(pack_path, "rb") as f:
                        self.current_pack = pickle.load(f)
                        self.current_pack_index += 1
                        self.cnt = 0  # Reset index for the new pack
                        failed_cnt = 0  # Reset failed count

                        # print(f"Unshuffled: {self.current_pack[:5]}")
                        # print(f"Unmatching: {self.unmatch_samples}")
                        if self.unmatch_samples:
                            # print(f"Shuffling pack {self.current_pack_index}")
                            audio_files = [sample[0] for sample in self.current_pack]
                            # print(f"Unshuffled: {audio_files[:5]}")
                            random.shuffle(audio_files)
                            # print(f"Shuffled: {audio_files[:5]}")
                            for i in range(len(self.current_pack)):
                                self.current_pack[i] = (
                                    audio_files[i],
                                    self.current_pack[i][1],
                                )
                            # print(f"Shuffled: {self.current_pack[:5]}")

                        break  # if successfully loaded, exit the loop
                else:
                    raise Exception(f"Pack file at path {pack_path} does not exist.")
            except:
                print(
                    f"Warning: pack {self.current_pack_index} seems to be corrupt. Skipping this pack."
                )
                failed_cnt += 1
                self.current_pack_index += 1  # move to next pack
        else:
            print(f"Warning: reached end of dataset. Resetting to first pack.")
            if failed_cnt == len(self.pack_filepaths):
                raise Exception("All packs seem to be corrupt. Aborting.")
            self.current_pack_index = 0
            self.load_pack()

    def __getitem__(self, index):
        # print(f"Getting item {index} from pack {self.current_pack_index}")
        # print(f"Current pack length: {len(self.current_pack)}")
        # print(f"Current pack index: {self.cnt}")
        if self.cnt >= len(self.current_pack):
            start_t = time.time()
            self.load_pack()
            print(f"Loading pack took {time.time() - start_t} seconds")
            self.cnt = 0  # Reset index for the new pack

        item = self.current_pack[self.cnt]
        self.cnt += 1
        print(f"text: {item[1]}")
        return item

    def __len__(self):
        """Get the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return self.num_files_per_pack * len(self.pack_filepaths)
