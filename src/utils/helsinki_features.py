import os
import glob
import numbers
from collections import OrderedDict
from typing import Union, List
from tqdm import tqdm

# from python_speech_features import fbank
import scipy.io.wavfile as wav
import numpy as np

from src.utils.text_processing import (
    read_lab_file,
    get_parts_from_lab_path,
    get_words_from_lab_lines,
)
from src.utils.prosody_tools.f0_processing import extract_f0, _interpolate


def sec_to_sample(sec, sr):
    """
    Transforms seconds to samples based on the sampling rate
    """
    return int(sec * sr)


def sample_to_sec(sample, sr):
    """
    Transforms samples to seconds based on the sampling rate
    """
    return sample / sr


def sec_to_index(sec, max_sec, arr_length):
    """
    Transforms seconds to an index in an array based on the maximum number of seconds and the array length
    """
    return int((sec / max_sec) * arr_length)


class QuestionExtractor:
    """
    Extracts the question from a lab file.
    """

    def __init__(self, lab_root):
        self.lab_root = lab_root

    def extract(self, lab_path):
        lab_file = read_lab_file(lab_path)
        question = lab_file[0].split(" ")[-1]
        return question


class WordBreakExtractor:
    """
    Extracts the word duration values from a wav snippet
    """

    def __init__(self, modes: Union[str, List[str]] = ["before", "after"]):
        """
        mode: duration. There is no other choice, we implement this for compatibility with e.g. F0Extractor
        """
        if isinstance(modes, str):
            modes = [modes]
        self.modes = modes

    def _extract_modes_from_duration(
        self, index, line_before=None, curr_line=None, line_after=None, is_last=False
    ):
        for mode in self.modes:
            if mode == "before":
                # if index == 1, then it's the pause before the utterance starts in the audio file
                return (
                    float(line_before[1]) - float(line_before[0])
                    if index != 1 and line_before and len(line_before) == 2
                    else 0.0
                )
            elif mode == "after":
                return (
                    float(line_after[1]) - float(line_after[0])
                    if line_after and len(line_after) == 2 and not is_last
                    else 0.0
                )
            else:
                raise ValueError("Invalid mode: {}".format(mode))

    def extract_from_lab_lines(self, waveform=None, lines=None, rate=None):
        """
        Extract the features for each word in the lines
        ::param waveform: the waveform of the whole file, not used here, just for compatibility
        ::param lines: a list of lab lines of format [start, end, word]
        ::param rate: sampling frequency, not used here, just for compatibility
        """

        features = list()
        for i, line in enumerate(lines):
            if len(line) < 3:
                continue

            # start_time = float(line[0])
            # end_time = float(line[1])
            word = line[2]
            line_before = (
                lines[lines.index(line) - 1] if lines.index(line) > 0 else None
            )
            curr_line = line
            line_after = (
                lines[lines.index(line) + 1]
                if lines.index(line) < len(lines) - 1
                else None
            )

            if word == "<unk>":
                # here the alignment failed, we do not want to use this sample
                return None

            # word_duration = end_time - start_time
            word_features = self._extract_modes_from_duration(
                index=i,
                line_before=line_before,
                curr_line=curr_line,
                line_after=line_after,
                is_last=(i == len(lines) - 1)
                or (i == len(lines) - 2 and len(line_after) == 2),
            )
            features.append(word_features)

        return features


class DurationExtractor:
    """
    Extracts the word duration values from a wav snippet
    """

    def __init__(self, modes: Union[str, List[str]] = "duration"):
        """
        mode: duration. There is no other choice, we implement this for compatibility with e.g. F0Extractor
        """
        if isinstance(modes, str):
            modes = [modes]
        self.modes = modes

    def _extract_modes_from_duration(self, word, duration):
        result = {}
        for mode in self.modes:
            if mode == "duration":
                result[mode] = duration
            elif mode == "duration_scaled":
                result[mode] = duration / len(word)
            else:
                raise ValueError("Invalid mode: {}".format(mode))

        return result

    def extract_from_lab_lines(self, waveform=None, lines=None, rate=None):
        """
        Extract the features for each word in the lines
        ::param waveform: the waveform of the whole file, not used here, just for compatibility
        ::param lines: a list of lab lines of format [start, end, word]
        ::param rate: sampling frequency, not used here, just for compatibility
        """

        features = list()
        for line in lines:
            if len(line) < 3:
                continue

            start_time = float(line[0])
            end_time = float(line[1])
            word = line[2]

            if word == "<unk>":
                # here the alignment failed, we do not want to use this sample
                return None

            word_duration = end_time - start_time
            word_features = self._extract_modes_from_duration(word, word_duration)
            features.append((word, word_features))

        return features


class F0Extractor:
    """
    Extracts the f0 values from a wav snippet
    """

    def __init__(
        self,
        modes: Union[str, List[str]] = "mean",
        interpolate: bool = False,
        keep_unvoiced: bool = True,
    ):
        """
        mode: mean, min, max, curve (all values)
        """
        if isinstance(modes, str):
            modes = [modes]
        self.modes = modes
        self.interpolate = interpolate
        self.keep_unvoiced = keep_unvoiced

    def _extract_modes_from_f0(self, f0):
        result = {}
        for mode in self.modes:
            if mode == "mean":
                result[mode] = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else None
            elif mode == "min":
                result[mode] = np.min(f0[f0 > 0]) if np.any(f0 > 0) else None
            elif mode == "max":
                result[mode] = np.max(f0[f0 > 0]) if np.any(f0 > 0) else None
            elif mode == "curve":
                result[mode] = f0
            else:
                raise ValueError("Invalid mode: {}".format(mode))
        return result

    def extract_from_lab_lines(
        self,
        waveform,
        lines,
        fs=16000,
        f0_min=30,
        f0_max=550,
        harmonics=10.0,
        voicing=50.0,
        configuration="pitch_tracker",
    ):
        """
        Extract the features for each word in the waveform
        ::param waveform: the waveform of the whole file
        ::param lines: a list of lab lines of format [start, end, word
        ::param fs: sampling frequency
        ::param f0_min: minimum f0
        ::param f0_max: maximum f0
        ::param harmonics: number of harmonics
        ::param voicing: voicing threshold
        ::param configuration: pitch tracker configuration
        """

        f0 = extract_f0(
            waveform=waveform,
            fs=fs,
            f0_min=f0_min,
            f0_max=f0_max,
            harmonics=harmonics,
            voicing=voicing,
            configuration=configuration,
        )
        # print(f"f0 shape: {f0.shape}")

        if self.interpolate:
            # print(f"interpolating f0")
            f0 = _interpolate(f0)

        max_time = float(lines[-1][1])

        features = list()
        for line in lines:
            if len(line) < 3:
                continue

            start_time = float(line[0])
            end_time = float(line[1])
            word = line[2]
            if word == "<unk>":
                return None

            start_idx = sec_to_index(start_time, max_time, f0.shape[0])
            end_idx = sec_to_index(end_time, max_time, f0.shape[0])
            # print(
            #     f"word: {word}, start: {start_time}, end: {end_time}, start_idx: {start_idx}, end_idx: {end_idx}"
            # )
            word_f0 = f0[start_idx:end_idx]
            # use only valid f0 values
            if not self.keep_unvoiced:
                word_f0 = word_f0[word_f0 > 0]
            # print(f"word_f0 shape: {word_f0.shape}")
            word_features = self._extract_modes_from_f0(word_f0)
            features.append((word, word_features))
        return features

    def extract(
        self,
        waveform,
        fs=16000,
        f0_min=30,
        f0_max=550,
        harmonics=10.0,
        voicing=50.0,
        configuration="pitch_tracker",
        interpolate=False,
    ):
        """
        Extract the features for the whole waveform
        ::param waveform: the waveform
        ::param fs: sampling frequency
        ::param f0_min: minimum f0
        ::param f0_max: maximum f0
        ::param harmonics: number of harmonics
        ::param voicing: voicing threshold
        ::param configuration: pitch tracker configuration
        """
        f0 = extract_f0(
            waveform=waveform,
            fs=fs,
            f0_min=f0_min,
            f0_max=f0_max,
            harmonics=harmonics,
            voicing=voicing,
            configuration=configuration,
        )
        if interpolate:
            f0 = _interpolate(f0)
        return self._extract_modes_from_f0(f0)


def get_features_from_lab_wav_path(
    lab_path: str, feature_extractor, wav_path: str = None
):
    """
    Get the features from a single file (lab and wav).
    ::param lab_path: path to the lab file
    ::param feature_extractor: function that takes a signal (snippet) and parameters and returns features
    ::param feature_extractor_params: parameters for the corresponding feature_extractor
    ::param wav_path: path to the wav file, optional
    """
    lines = read_lab_file(
        lab_path
    )  # list of lists, where each list contains the start time, end time, and word/phoneme

    if wav_path is not None:
        (rate, sig) = wav.read(wav_path)
        features = feature_extractor.extract_from_lab_lines(sig, lines, rate)
    else:
        features = feature_extractor.extract_from_lab_lines(None, lines, None)

    return features


def get_duration_features_from_lab_root(
    lab_root: str,
    wav_root: str,
    feature_extractor,
    txt_type: str = "original",
    verbose: bool = False,
):
    """
    Get the features from all lab files recursively traversing from a root directory.
    ::param lab_root: path to the directory containing the lab files
    ::param wav_root: path to the directory containing the wav files
    ::param txt_type: 'original' or 'normalized'
    Returns a list of dictionaries, where each dictionary one sentence contains the words and their corresponding features.
    """
    # TODO: pass a list of feature extractors, and normalize their features separately

    all_samples = []
    failed_alignments = 0
    failed_lengths = []

    for reader in tqdm(os.listdir(lab_root), desc="Readers"):
        if reader == ".DS_Store":
            continue

        reader_path = os.path.join(lab_root, reader)
        # max_energy = -1  # to later normalize energy of each reader
        # min_energy = np.inf  # to later normalize energy of each reader
        reader_samples = []
        # print(f"reader {reader}")

        for book in os.listdir(reader_path):
            if book == ".DS_Store":
                continue
            book_path = os.path.join(reader_path, book)
            # print(f"book {book}")
            book_files = [file for file in os.listdir(book_path) if file != ".DS_Store"]

            for ut in book_files:
                # print("utterance", ut)
                if ut == ".DS_Store":
                    continue
                ut_path_lab = os.path.join(book_path, ut)
                # print("utterance lab path ", ut_path_lab)
                ut_path_wav = os.path.join(
                    wav_root, reader, book, ut.replace(".lab", ".wav")
                )
                # print("utterance wav path ", ut_path_wav)
                ut_path_txt = os.path.join(
                    wav_root, reader, book, ut.replace(".lab", ".original.txt")
                )
                if verbose:
                    print(f"Processing file {ut_path_lab}")
                # text = str(open(ut_path_txt).read())
                features = get_features_from_lab_wav_path(
                    lab_path=ut_path_lab,
                    feature_extractor=feature_extractor,
                    wav_path=ut_path_wav,
                )

                if features is None:
                    failed_alignments += 1
                    continue

                if verbose:
                    print(f"Extracted numer of features: {len(features)}")

                text = [item[0] for item in features]

                sample = OrderedDict()
                sample["reader"] = reader
                sample["book"] = book
                sample["text"] = text
                sample["features"] = features
                sample["path_txt"] = ut_path_txt
                sample["path_lab"] = ut_path_lab
                sample["path_wav"] = ut_path_wav
                sample["filename"] = ut.replace(".lab", "")
                # print(sample)
                reader_samples.append(sample)
        # add the sentence to the list of all sentences
        all_samples += reader_samples

    if verbose:
        print("Failed alignments: ", failed_alignments)
        print("Total aligned sentences: ", len(all_samples))

    return all_samples


def get_features_from_lab_root(
    lab_root: str,
    wav_root: str,
    feature_extractor,
    txt_type: str = "original",
    verbose: bool = False,
):
    """
    Get the features from all lab files recursively traversing from a root directory.
    ::param lab_root: path to the directory containing the lab files
    ::param wav_root: path to the directory containing the wav files
    ::param txt_type: 'original' or 'normalized'
    Returns a list of dictionaries, where each dictionary one sentence contains the words and their corresponding features.
    """
    # TODO: pass a list of feature extractors, and normalize their features separately

    all_samples = []
    failed_alignments = 0
    failed_lengths = []

    for reader in tqdm(os.listdir(lab_root), desc="Readers"):
        if reader == ".DS_Store":
            continue

        reader_path = os.path.join(lab_root, reader)
        # max_energy = -1  # to later normalize energy of each reader
        # min_energy = np.inf  # to later normalize energy of each reader
        reader_samples = []
        reader_f0_values = []  # to later normalize f0 of each reader
        # print(f"reader {reader}")

        for book in os.listdir(reader_path):
            if book == ".DS_Store":
                continue
            book_path = os.path.join(reader_path, book)
            # print(f"book {book}")
            book_files = [file for file in os.listdir(book_path) if file != ".DS_Store"]

            for ut in book_files:
                # print("utterance", ut)
                if ut == ".DS_Store":
                    continue
                ut_path_lab = os.path.join(book_path, ut)
                # print("utterance lab path ", ut_path_lab)
                ut_path_wav = os.path.join(
                    wav_root, reader, book, ut.replace(".lab", ".wav")
                )
                # print("utterance wav path ", ut_path_wav)
                ut_path_txt = os.path.join(
                    wav_root, reader, book, ut.replace(".lab", ".original.txt")
                )
                if verbose:
                    print(f"Processing file {ut_path_lab}")
                # text = str(open(ut_path_txt).read())
                features = get_features_from_lab_wav_path(
                    lab_path=ut_path_lab,
                    feature_extractor=feature_extractor,
                    wav_path=ut_path_wav,
                )

                if features is None:
                    failed_alignments += 1
                    continue

                if verbose:
                    print(f"Extracted numer of features: {len(features)}")

                text = [item[0] for item in features]

                reader_f0_values.extend(
                    np.concatenate([item[1]["curve"] for item in features])
                )

                sample = OrderedDict()
                sample["reader"] = reader
                sample["book"] = book
                sample["text"] = text
                sample["features"] = features
                sample["path_txt"] = ut_path_txt
                sample["path_lab"] = ut_path_lab
                sample["path_wav"] = ut_path_wav
                sample["filename"] = ut.replace(".lab", "")
                # print(sample)
                reader_samples.append(sample)

        # Z-score normalization of f0
        reader_f0_values = np.array(reader_f0_values)
        reader_mean_f0 = np.mean(reader_f0_values)
        reader_std_f0 = np.std(reader_f0_values)

        # print("reader samples", reader_samples)

        for sample in reader_samples:
            # print(sample)
            for word, features in sample["features"]:
                # if curve is all 0, we cannot compute any statistics
                if np.all(features["curve"] == 0):
                    features["mean"] = None
                    features["max"] = None
                    features["min"] = None
                    # features["curve"] = np.array(
                    #     features["curve"]
                    # )
                    continue

                # Z-score normalization of values that are not 0
                features["curve"] = np.array(features["curve"])
                # print(f"word {word} curve {sample['features'][word]['curve']}")
                # normalize only the voiced parts
                features["curve"][features["curve"] > 0] = (
                    features["curve"][features["curve"] > 0] - reader_mean_f0
                ) / reader_std_f0
                # print(
                #     f"after norm word {word} curve {sample['features'][word]['curve']}"
                # )

                # Mean of normalized values (None was checked before)
                features["mean"] = np.mean(features["curve"])
                # print(f"word {word} mean {sample['features'][word]['mean']}")

                # max of normalized values (None was checked before)
                features["max"] = np.max(features["curve"])
                # print(f"word {word} max {sample['features'][word]['max']}")

                # min of normalized values (None was checked before)
                features["min"] = np.min(features["curve"])
                # print(f"word {word} min {sample['features'][word]['min']}")

        # print("reader samples after normalization", reader_samples)

        # add the sentence to the list of all sentences
        all_samples += reader_samples

    if verbose:
        print("Failed alignments: ", failed_alignments)
        print("Total aligned sentences: ", len(all_samples))
        # print(
        #     "Mean sentence length: ",
        #     np.mean([len(s["words_to_features"]) for s in all_samples]),
        # )
        # print(
        #     "Standard deviation sentence length: ",
        #     np.std([len(s["words_to_features"]) for s in all_samples]),
        # )
        # print("Mean failed sentence length: ", np.mean(failed_lengths))
        # print("Standard deviation failed sentence length: ", np.std(failed_lengths))

    return all_samples


# TODO: merge this into one generally applicable function
def get_duration_features_from_lab_root(
    lab_root: str,
    wav_root: str,
    feature_extractor,
    txt_type: str = "original",
    verbose: bool = False,
):
    """
    Get the features from all lab files recursively traversing from a root directory.
    ::param lab_root: path to the directory containing the lab files
    ::param wav_root: path to the directory containing the wav files
    ::param txt_type: 'original' or 'normalized'
    Returns a list of dictionaries, where each dictionary one sentence contains the words and their corresponding features.
    """
    # TODO: pass a list of feature extractors, and normalize their features separately

    all_samples = []
    failed_alignments = 0
    failed_lengths = []

    for reader in tqdm(os.listdir(lab_root), desc="Readers"):
        if reader == ".DS_Store":
            continue

        reader_path = os.path.join(lab_root, reader)
        # max_energy = -1  # to later normalize energy of each reader
        # min_energy = np.inf  # to later normalize energy of each reader
        reader_samples = []
        reader_f0_values = []  # to later normalize f0 of each reader
        # print(f"reader {reader}")

        for book in os.listdir(reader_path):
            if book == ".DS_Store":
                continue
            book_path = os.path.join(reader_path, book)
            # print(f"book {book}")
            book_files = [file for file in os.listdir(book_path) if file != ".DS_Store"]

            for ut in book_files:
                # print("utterance", ut)
                if ut == ".DS_Store":
                    continue
                ut_path_lab = os.path.join(book_path, ut)
                # print("utterance lab path ", ut_path_lab)
                ut_path_wav = os.path.join(
                    wav_root, reader, book, ut.replace(".lab", ".wav")
                )
                # print("utterance wav path ", ut_path_wav)
                ut_path_txt = os.path.join(
                    wav_root, reader, book, ut.replace(".lab", ".original.txt")
                )
                if verbose:
                    print(f"Processing file {ut_path_lab}")
                # text = str(open(ut_path_txt).read())
                features = get_features_from_lab_wav_path(
                    lab_path=ut_path_lab,
                    feature_extractor=feature_extractor,
                    # wav_path=ut_path_wav,
                )

                if features is None:
                    failed_alignments += 1
                    continue

                if verbose:
                    print(f"Extracted numer of features: {len(features)}")

                text = [item[0] for item in features]

                sample = OrderedDict()
                sample["reader"] = reader
                sample["book"] = book
                sample["text"] = text
                sample["features"] = features
                sample["path_txt"] = ut_path_txt
                sample["path_lab"] = ut_path_lab
                sample["path_wav"] = ut_path_wav
                sample["filename"] = ut.replace(".lab", "")
                # print(sample)
                reader_samples.append(sample)
        all_samples += reader_samples

    if verbose:
        print("Failed alignments: ", failed_alignments)
        print("Total aligned sentences: ", len(all_samples))

    return all_samples


# def get_sample_paths(
#     lab_root: str,
#     wav_root: str,
#     subsets: list,
#     txt_type: str = "original",
#     verbose: bool = False,
# ):
#     """
#     Get the features from all lab files recursively traversing from a root directory.
#     ::param lab_root: path to the directory containing the lab files
#     ::param wav_root: path to the directory containing the wav files
#     ::param txt_type: 'original' or 'normalized'
#     Returns a list of dictionaries, where each dictionary one sentence contains the words and their corresponding features.
#     """
#     utterances = []

#     # iterate over dirs in lab_root
#     for subset in subsets:
#         if not os.path.isdir(os.path.join(lab_root, subset)):
#             continue  # continue if not a dir
#         if verbose:
#             print(f"subset {subset}")
#         subset_path = os.path.join(lab_root, subset)

#         for reader in os.listdir(subset_path):
#             if verbose:
#                 print(f"reader {reader}")
#             if reader == ".DS_Store":
#                 continue
#             reader_path = os.path.join(subset_path, reader)

#             for book in os.listdir(reader_path):
#                 if book == ".DS_Store":
#                     continue
#                 if verbose:
#                     print(f"book {book}")
#                 book_path = os.path.join(reader_path, book)
#                 book_files = [
#                     file for file in os.listdir(book_path) if file != ".DS_Store"
#                 ]

#                 for ut in book_files:
#                     # print("utterance", ut)
#                     if ut == ".DS_Store":
#                         continue
#                     ut_path_lab = os.path.join(book_path, ut)
#                     # print("utterance lab path ", ut_path_lab)
#                     ut_path_wav = os.path.join(
#                         wav_root, subset, reader, book, ut.replace(".lab", ".wav")
#                     )
#                     # print("utterance wav path ", ut_path_wav)
#                     ut_path_txt = os.path.join(
#                         wav_root,
#                         subset,
#                         reader,
#                         book,
#                         ut.replace(".lab", ".original.txt"),
#                     )
#                     ut_path_norm_txt = os.path.join(
#                         wav_root,
#                         subset,
#                         reader,
#                         book,
#                         ut.replace(".lab", ".normalized.txt"),
#                     )
#                     text = str(open(ut_path_txt).read())

#                     sample = OrderedDict()
#                     sample["reader_id"] = reader
#                     sample["text"] = text
#                     # sample['words_to_features'] = dict(zip(words, labels))
#                     sample["path_original_txt"] = ut_path_txt
#                     sample["path_normalized_txt"] = ut_path_norm_txt
#                     sample["path_lab"] = ut_path_lab
#                     sample["path_wav"] = ut_path_wav
#                     utterances.append(sample)

#     return utterances


def utterance_to_question_classification(utterances):
    """
    Takes an utterance of the following format:
    OrderedDict([('reader_id', '237'),
             ('text', 'Yes, it must be confessed.'),
             ('path_original_txt',
              '/Users/lukas/Desktop/projects/MIT/data/LibriTTS/debug/237/126133/237_126133_000002_000000.original.txt'),
             ('path_normalized_txt',
              '/Users/lukas/Desktop/projects/MIT/data/LibriTTS/debug/237/126133/237_126133_000002_000000.normalized.txt'),
             ('path_lab',
              '/Users/lukas/Desktop/projects/MIT/data/LibriTTSCorpusLabel/debug/237/126133/237_126133_000002_000000.lab'),
             ('path_wav',
              '/Users/lukas/Desktop/projects/MIT/data/LibriTTS/debug/237/126133/237_126133_000002_000000.wav')])
    and returns the following format as lists:
    class 1: question
    class 0: not question
    ['Yes it must be confessed'], [0]
    """
    texts_with_punctuation_removed = []
    labels = []
    for ut in utterances:
        text = ut["text"]
        lab_path = ut["path_lab"]
        lines = read_lab_file(lab_path)
        words = get_words_from_lab_lines(lines)
        texts_with_punctuation_removed.append(" ".join(words))
        # if text ends with question mark, 1 else 0
        labels.append(1 if text[-1] == "?" else 0)
    return texts_with_punctuation_removed, labels
