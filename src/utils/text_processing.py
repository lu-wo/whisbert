import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import string
from typing import Union, List
import nltk
import re
import glob
import os
from nltk.corpus import cmudict
import syllables
import pyphen
from g2p_en import G2p


dic = pyphen.Pyphen(lang="en")
g2p = G2p()

try:
    d = cmudict.dict()
except:
    print("Downloading nltk data...")
    nltk.download("cmudict")
    d = cmudict.dict()

import os
import glob
import pandas as pd


def create_audio_alignment_mapping_file(audio_root: str, alignment_root: str):
    mapping = []
    audio_files = glob.glob(os.path.join(audio_root, "**", "*.flac"), recursive=True)

    for audio_file in audio_files:
        audio_basename = os.path.basename(audio_file).rsplit(".", 1)[0]
        transcript_files = glob.glob(
            os.path.join(alignment_root, "**", "*.txt"), recursive=True
        )

        for transcript_file in transcript_files:
            transcript_basename = os.path.basename(transcript_file).rsplit(".", 1)[0]
            if audio_basename in transcript_basename:
                mapping.append(dict(audio=audio_file, transcript=transcript_file))

    df = pd.DataFrame(mapping)
    print(f"Found: {len(df)}")
    return df


def create_audio_alignment_mapping(audio_root: str, alignment_root: str):
    found = 0
    not_found = 0
    mapping = []

    # Get all .flac files in the audio_root directory
    audio_files = glob.glob(f"{audio_root}/**/*.flac", recursive=True)

    for audio_file in audio_files:
        # Get the name of the file without the extension
        name = os.path.basename(audio_file).rsplit(".", 1)[0]

        # Construct the path to the expected alignment directory
        alignment_dir = os.path.join(alignment_root, name)

        # Check if the alignment directory exists
        if os.path.isdir(alignment_dir):
            # Get the transcript file in the alignment directory
            transcript_files = glob.glob(f"{alignment_dir}/*.txt")
            if transcript_files:
                # Append the audio file and transcript file to the mapping
                mapping.append(dict(audio=audio_file, transcript=transcript_files[0]))
                found += 1
            else:
                not_found += 1
        else:
            not_found += 1

    print(f"Found: {found}")
    print(f"Not found: {not_found}")

    # Convert mapping to a pandas DataFrame
    df = pd.DataFrame(mapping)

    return df


class TextFileProcessor:
    def __init__(self, root_dirs, max_words=80, min_confidence=0.0):
        self.root_dir = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.max_words = max_words
        self.min_confidence = min_confidence
        self.data, self.files = self._process_files()

    def _process_files(self):
        data = {}
        file_paths = []
        for root_dir in self.root_dir:
            for file_path in glob.glob(
                os.path.join(root_dir, "**", "*.txt"), recursive=True
            ):
                file_name = os.path.basename(file_path)
                data[file_name] = read_peoples_speech_txtfile(file_path)
                file_paths.append(file_path)

        return data, file_paths

    def create_text_samples(self):
        samples = []
        for file_data in self.data.values():
            sample = []
            for word_data in file_data:
                if word_data["confidence"] < self.min_confidence:
                    break  # skip this file
                sample.append(word_data["word"])
                if word_data["word"].endswith(".") or len(sample) >= self.max_words:
                    samples.append(" ".join(sample))
                    sample = []
        return samples

    def count_words_in_samples(self, samples):
        return sum(len(sample.split()) for sample in samples)

    def split_samples(
        self, samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0
    ):
        random.seed(seed)
        random.shuffle(samples)
        train_end = int(len(samples) * train_ratio)
        val_end = train_end + int(len(samples) * val_ratio)
        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]
        return train_samples, val_samples, test_samples


def read_peoples_speech_txtfile(file_path):
    """
    Format: (start end word confidence missing)
    0.028 0.588 So 0.517 missing
    ...
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    result = []
    prev_end = 0.0  # end time of the previous word, initialized to 0
    for line in lines:
        elements = line.strip().split()
        try:
            start = float(elements[0])
            end = float(elements[1])
            word = elements[2]
            confidence = float(elements[3])
        except:
            # found sth like "missing missing missing missing"
            # TODO: might handle this better
            continue
        break_before = start - prev_end  # calculate the break before the current word
        duration = end - start  # calculate the duration of the word
        prev_end = end  # update the end time of the previous word
        result.append(
            {
                "start": start,
                "end": end,
                "word": word,
                "confidence": confidence,
                "break_before": break_before,
                "duration": duration,
            }
        )

    return result


def find_stress_syllable_start(
    syllables, stress_index, phoneme_lab_lines, word_start, word_end, verbose=False
):
    phoneme_lab_lines = [
        line for line in phoneme_lab_lines if len(line) > 2
    ]  # Remove empty lines

    # Find the lines that are in the range
    phoneme_lab_lines = [
        (float(start), float(end), phoneme)
        for start, end, phoneme in phoneme_lab_lines
        if start >= word_start and end <= word_end
    ]

    if verbose:
        print("phoneme lab lines", phoneme_lab_lines)

    # Extract the stress syllable phonemes
    stress_syllable = syllables[stress_index]
    if verbose:
        print(f"Syllables: {syllables}")
        print(f"stress syllable: {stress_syllable} at index {stress_index}")

    candidates = []

    # The current window of phonemes
    window_phonemes = []

    for start, end, phoneme in phoneme_lab_lines:
        # print(start, end, phoneme)
        window_phonemes.append((start, phoneme))

        # Build the current window string
        curr_str = "".join(p for _, p in window_phonemes)

        if curr_str == stress_syllable:
            # If the window matches the stress syllable, add the start time as a candidate
            candidates.append(window_phonemes[0][0])

        # If the window is larger than the stress syllable, remove phonemes from the start
        while len(curr_str) > len(stress_syllable):
            window_phonemes.pop(0)
            curr_str = "".join(p for _, p in window_phonemes)

        # Check if the window matches the stress syllable again after removing phonemes
        if curr_str == stress_syllable:
            candidates.append(window_phonemes[0][0])

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 0:
        return None  # no candidates found

    # remove ambiguity by using stress index
    try:
        return candidates[stress_index]  # stress index too big
    except IndexError:
        return candidates[-1]  # so let's just take the last one as approx


def extract_phonemes(
    phoneme_lab_lines, word_phonemes, start_time, end_time, verbose=False
):
    # Split the file content by lines and split each line by tabs
    # Clean the phoneme lines by removing empty strings
    phoneme_lab_lines = [
        line for line in phoneme_lab_lines if len(line) > 2
    ]  # Remove empty lines
    # Convert start times, end times, and phonemes to a list of tuples
    phonemes = [
        (float(start), float(end), phoneme) for start, end, phoneme in phoneme_lab_lines
    ]

    if verbose:
        print(phonemes)

    # Get the phonemes for the word of interest within the start and end times
    word_phonemes_data = [
        (start, end, phoneme)
        for start, end, phoneme in phonemes
        if start_time <= start <= end_time and phoneme in word_phonemes
    ]

    return word_phonemes_data


def nb_syllables(word):
    """
    Returns the number of syllables in a word.
    If the word is not in the CMU Pronouncing Dictionary, use syllables as a fallback.
    """
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except KeyError:
        return syllables.estimate(word)


def syllabify(word):
    """
    Syllabifies a word using the CMU Pronouncing Dictionary.
    If the word is not in the dictionary, use g2p_en as a fallback.
    Returns: a list of syllables
    """
    try:
        # Get the syllabified phonemes from the dictionary
        syllabified_phonemes = d[word.lower()][0]

        # Create syllables from the phonemes
        syllables = []
        syllable = ""

        for phoneme in syllabified_phonemes:
            # Phonemes with numbers are the end of a syllable
            if "0" in phoneme or "1" in phoneme or "2" in phoneme:
                syllable += phoneme
                syllables.append(syllable)
                syllable = ""
            else:
                syllable += phoneme

        # Catch any remaining phonemes as a syllable
        if syllable:
            syllables.append(syllable)

        return syllables

    except KeyError:
        print(
            f"Word '{word}' not in CMU Pronouncing Dictionary. Using g2p for ARPABET conversion."
        )
        # Use g2p_en as a fallback
        arpabet_phonemes = g2p(word)
        syllables = []
        syllable = ""

        for phoneme in arpabet_phonemes:
            if "0" in phoneme or "1" in phoneme or "2" in phoneme:
                syllable += phoneme
                syllables.append(syllable)
                syllable = ""
            else:
                syllable += phoneme

        # Catch any remaining phonemes as a syllable
        if syllable:
            syllables.append(syllable)

        return syllables


class CelexReader:
    def __init__(self, file_path):
        self.data = self._load_data(file_path)

    def _load_data(self, file_path):
        # Dictionary to store the information
        data = {}

        # Open the file and read line by line
        with open(file_path, "r") as file:
            # Skipping header line (assuming the first line is the header)
            # If there's no header, comment the line below
            # next(file)

            # Reading each line
            for line in file:
                # Splitting the line by tabs
                (
                    head,
                    cls,
                    strs_pat,
                    phon_syl_disc,
                    morph_status,
                    cob,
                ) = line.strip().split("\\")

                # Creating a dictionary for the current word
                info = {
                    "Class": cls,
                    "StressPattern": strs_pat,
                    "PhoneticSyllable": phon_syl_disc,
                    "MorphStatus": morph_status,
                    "Frequency": cob,
                }

                # Adding to the data dictionary
                data[head] = info

        return data

    def lookup(self, word):
        # Returning the information for the requested word
        return self.data.get(word, None)

    def get_stress_syllable(self, word):
        return self.lookup(word).get("StressPattern", None)

    def get_class(self, word):
        return self.lookup(word).get("Class", None)

    def get_phonetic_syllable(self, word):
        return self.lookup(word).get("PhoneticSyllable", None)

    def get_morph_status(self, word):
        return self.lookup(word).get("MorphStatus", None)

    def get_frequency(self, word):
        return self.lookup(word).get("Frequency", None)

    def get_stress_position(self, word):
        try:
            stress_syllable = self.get_stress_syllable(word)
            if stress_syllable is not None:
                start = stress_syllable.find("1") / len(stress_syllable)
                end = (stress_syllable.rfind("1") + 1) / len(stress_syllable)
                return start, end
        except:
            return 0, 1  # default to full word stressed

    def get_stress_index(self, word):
        try:
            stress_syllable = self.get_stress_syllable(word)
            if stress_syllable is not None:
                return stress_syllable.find("1")
        except:
            return 0


def get_wordlist_from_string(string: str) -> List[str]:
    return re.findall(r"\w+'?\w*", string)


def get_part_of_speech(word):
    global nltk_downloads_complete
    if not nltk_downloads_complete:
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        nltk_downloads_complete = True

    tokens = nltk.word_tokenize(word)
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags[0][1]


class WordRanking:
    def __init__(self, file_path):
        self.rank_data = {}
        self._read_file(file_path)

    def _read_file(self, file_path):
        with open(file_path, "r") as file:
            next(file)  # Skip header line
            for line in file:
                parts = line.split()
                rank = int(parts[0])
                word = parts[1]
                count = int(parts[2].replace(",", ""))
                percent = float(parts[3].strip("%"))
                cumulative = float(parts[4].strip("%"))

                self.rank_data[word] = {
                    "rank": rank,
                    "count": count,
                    "percent": percent,
                    "cumulative": cumulative,
                }

    def get_rank(self, word):
        word = word.lower()
        if word in self.rank_data:
            return self.rank_data[word]["rank"]
        return None

    def get_word(self, rank):
        for word, data in self.rank_data.items():
            if data["rank"] == rank:
                return word
        return None

    def get_count(self, word):
        word = word.lower()
        if word in self.rank_data:
            return self.rank_data[word]["count"]
        return None

    def get_percent(self, word):
        word = word.lower()
        if word in self.rank_data:
            return self.rank_data[word]["percent"]
        return None

    def get_cumulative(self, word):
        word = word.lower()
        if word in self.rank_data:
            return self.rank_data[word]["cumulative"]
        return None

    def is_in_top_100k(self, word):
        word = word.lower()
        if word in self.rank_data and self.rank_data[word]["rank"] <= 100000:
            return True
        return False

    def is_in_top_Xk(self, word, X=10):
        word = word.lower()
        if word in self.rank_data and self.rank_data[word]["rank"] <= X * 1000:
            return True
        return False


def python_remove_whitespace(string):
    return "".join(string.split())


def python_lowercase_remove_punctuation(
    input_text: Union[str, List[str]]
) -> Union[str, List[str]]:
    if isinstance(input_text, str):
        return input_text.lower().translate(str.maketrans("", "", string.punctuation))
    elif isinstance(input_text, list):
        return [python_lowercase_remove_punctuation(text) for text in input_text]
    else:
        raise ValueError("Input must be a string or a list of strings")


def python_lowercase(input_text: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(input_text, str):
        return input_text.lower()
    elif isinstance(input_text, list):
        return [python_lowercase(text) for text in input_text]
    else:
        raise ValueError("Input must be a string or a list of strings")


def python_remove_punctuation(
    input_text: Union[str, List[str]]
) -> Union[str, List[str]]:
    if isinstance(input_text, str):
        return input_text.translate(str.maketrans("", "", string.punctuation))
    elif isinstance(input_text, list):
        return [python_remove_punctuation(text) for text in input_text]
    else:
        raise ValueError("Input must be a string or a list of strings")


def get_paths_from_root(root_dir, ends_with=".wav"):
    """
    Returns a list of paths to files in root_dir that end with ends_with.
    """
    paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(ends_with):
                paths.append(os.path.join(root, file))
    return paths


def read_lab_file(lab_path):
    """
    Returns a list of lists, where each list lines[i] contains the start time (lines[i][0]), end time (lines[i][1]), and word/phoneme (lines[i][2]).
    Note that if there a pause, len(lines[i]) < 3, since there is no word/phoneme
    """
    with open(lab_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split("\t") for line in lines]
    # Cast start and end to float
    for line in lines:
        line[0], line[1] = float(line[0]), float(line[1])
    return lines


def remove_breaks_from_lab_lines(lines):
    """
    Returns a list of lists, where each list lines[i] contains the start time (lines[i][0]), end time (lines[i][1]), and word/phoneme (lines[i][2]).
    Note that if there a pause, len(lines[i]) < 3, since there is no word/phoneme
    """
    return [line for line in lines if len(line) == 3]


def get_parts_from_lab_path(lab_path):
    """
    Returns the name parts of the lab file path.
    works for LibriTTS
    """
    path = lab_path.split(".")[0].split("/")[-1]
    reader, book, ut1, ut2 = path.split("_")
    return reader, book, ut1 + "_" + ut2


def get_words_from_lab_lines(lines):
    """
    Returns a list of words from the lab file lines.
    """
    words = []
    for line in lines:
        if len(line) == 3:
            words.append(line[2])
    return words
