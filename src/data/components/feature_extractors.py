import re
import pickle
import os
from collections import defaultdict
import numpy as np
import scipy.stats as stats
from scipy import signal
from tqdm import tqdm
from collections import OrderedDict
from scipy.fftpack import dct, fft
import concurrent.futures as cf

from src.utils.text_processing import find_stress_syllable_start
from src.utils.text_processing import syllabify
from src.utils.text_processing import CelexReader

from src.utils.text_processing import (
    python_lowercase,
    python_remove_punctuation,
    python_lowercase_remove_punctuation,
    read_lab_file,
    remove_breaks_from_lab_lines,
    nb_syllables,
)
from src.utils.helsinki_features import (
    get_features_from_lab_root,
    F0Extractor,
    DurationExtractor,
    WordBreakExtractor,
    read_lab_file,
    get_features_from_lab_wav_path,
    get_duration_features_from_lab_root,
    get_features_from_lab_root,
)
from src.utils.utils import min_length_of_lists, sec_to_idx, equal_length_or_none
from src.utils.prosody_tools.misc import read_wav, normalize_std
from src.utils.prosody_tools import (
    f0_processing,
    smooth_and_interp,
    energy_processing,
    duration_processing,
)
from src.utils.helsinki_features import WordBreakExtractor

INVALID_SYMBOLS = ["<unk>"]


class ProsodyFeatureExtractor:
    """
    This class uses the code from the wavelet_prosody_toolkit (https://github.com/asuni/wavelet_prosody_toolkit)
    in order to extract continuous prosodic features for:
    - f0 curves, parameterized
    - energy curves and mean energy per word
    - duration curves
    - prominence curves as composite signal of the three signals above
    """

    def __init__(
        self,
        lab_root: str = None,
        wav_root: str = None,
        phoneme_lab_root: str = None,
        data_cache: str = None,
        extract_f0: bool = False,
        f0_mode: str = "dct",
        f0_n_coeffs: int = 4,
        f0_stress_localizer: str = "celex",
        f0_window: int = 500,  # ms
        f0_resampling_length: int = 100,
        celex_path: str = None,
        extract_energy: bool = False,
        energy_mode: str = "mean",
        extract_word_duration: bool = False,
        word_duration_mode: str = "syllable_norm",
        extract_duration: bool = False,
        extract_pause_before: bool = False,
        extract_pause_after: bool = False,
        extract_prominence: bool = False,
        prominence_mode: str = "mean",
        f0_min: float = 50,
        f0_max: float = 400,
        f0_voicing: float = 50,
        energy_min_freq: float = 200,
        energy_max_freq: float = 5000,
        f0_weight: float = 1.0,
        energy_weight: float = 1.0,
        duration_weight: float = 0.5,
        unallowed_symbols: list = INVALID_SYMBOLS,
    ):
        """
        ::param lab_root: path to the directory containing the lab files
        ::param wav_root: path to the directory containing the wav files
        ::param phoneme_lab_root: path to the directory containing the phoneme lab files
        ::param data_cache: path to the directory where the cache is stored
        ::param extract_f0: whether to extract f0 features
        ::param f0_mode: how to parameterize the f0 curve choices: dct, fft, poly
        ::param f0_n_coeffs: number of coefficients to keep for dct and fft
        ::param f0_stress_localizer: how to parameterize the stress curve choices: celex, full_curve
        ::param f0_window: window size for around the stress localizer
        ::param extract_energy: whether to extract energy features
        ::param energy_mode: how to parameterize the energy curve choices: max, mean, curve
        ::param extract_word_duration: whether to extract duration features per word
        ::param word_duration_mode: how to parameterize the word duration curve choices: char_norm (divide by number of characters)
        ::param extract_duration: whether to extract duration features
        ::param extract_pause_before: whether to extract pause features before the word
        ::param extract_pause_after: whether to extract pause features after the word
        ::param extract_prominence: whether to extract prominence features (composite signal of f0, energy and duration)
        ::param prominence_mode: how to parameterize the prominence curve choices: max, mean, curve

        ::param prominence_mode: how to parameterize the prominence curve choices: max, mean
        """
        self.lab_root = lab_root
        self.wav_root = wav_root
        self.phoneme_lab_root = phoneme_lab_root
        self.data_cache = data_cache

        self.extract_f0 = extract_f0
        self.f0_mode = f0_mode
        self.f0_n_coeffs = f0_n_coeffs
        self.f0_stress_localizer = f0_stress_localizer
        self.f0_window = f0_window
        self.f0_resampling_length = f0_resampling_length
        if self.extract_f0:
            self.celex_path = celex_path
            self.celex_manager = CelexReader(celex_path)
        self.extract_energy = extract_energy
        self.energy_mode = energy_mode
        self.extract_word_duration = extract_word_duration
        self.word_duration_mode = word_duration_mode
        self.extract_duration = extract_duration
        self.extract_pause_before = extract_pause_before
        if self.extract_pause_before:
            self.pause_before_extractor = WordBreakExtractor(modes="before")
        self.extract_pause_after = extract_pause_after
        if self.extract_pause_after:
            self.pause_after_extractor = WordBreakExtractor(modes="after")
        self.extract_prominence = extract_prominence
        if self.extract_prominence:
            self.extract_f0 = True
            self.extract_energy = True
            self.extract_duration = True
        self.prominence_mode = prominence_mode

        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_voicing = f0_voicing
        self.energy_min_freq = energy_min_freq
        self.energy_max_freq = energy_max_freq

        self.f0_weight = f0_weight
        self.energy_weight = energy_weight
        self.duration_weight = duration_weight

        self.unallowed_symbols = unallowed_symbols

        # print what is going to be extracted
        extracted_features = []
        for feature in [
            "f0",
            "energy",
            "word_duration",
            "duration",
            "pause_before",
            "pause_after",
            "prominence",
        ]:
            if getattr(self, f"extract_{feature}"):
                extracted_features.append(feature)
        print(f"Extracted features ----: {extracted_features}")

        # process files
        self.process_files()

    def _parameterize_f0(self, f0):
        if self.f0_mode == "dct":
            f0_coeffs = dct(f0, type=2, norm="ortho")
        elif self.f0_mode == "fft":
            f0_coeffs = fft(f0)
        elif self.f0_mode == "poly":
            degree = self.f0_n_coeffs - 1  # Degree of polynomial
            f0_coeffs = np.polyfit(np.arange(len(f0)), f0, degree)
        else:
            raise ValueError("Unknown f0_mode: {}".format(self.f0_mode))

        return f0_coeffs[: self.f0_n_coeffs]

    def _extract_f0(self, waveform, fs):
        f0_raw = f0_processing.extract_f0(
            waveform=waveform,
            fs=fs,
            f0_min=self.f0_min,
            f0_max=self.f0_max,
            voicing=self.f0_voicing,
        )
        f0_interpolated = f0_processing.process(f0_raw)
        f0_normalized = normalize_std(f0_interpolated)
        return f0_normalized

    def _extract_energy(self, waveform, fs):
        energy = energy_processing.extract_energy(
            waveform=waveform,
            fs=fs,
            min_freq=self.energy_min_freq,
            max_freq=self.energy_max_freq,
            method="rms",
        )
        energy_smooth = smooth_and_interp.peak_smooth(
            energy, 30, 3
        )  # 30, 3 are fixed in helsinki code
        energy_normalized = normalize_std(energy_smooth)
        return energy_normalized

    def _extract_duration(self, waveform, fs, lab_lines, resample_length):
        duration_signal = duration_processing.duration(lab_lines, rate=fs)
        duration_norm = normalize_std(duration_signal)
        duration_norm = signal.resample(duration_norm, resample_length)
        return duration_norm

    def _extract_prominence(self, f0, energy, duration):
        f0 = f0[: min(len(f0), len(energy))]
        energy = energy[: min(len(f0), len(energy))]
        duration = duration[: min(len(f0), len(energy))]

        prominence = (
            self.f0_weight * f0
            + self.energy_weight * energy
            + self.duration_weight * duration
        )
        prominence = smooth_and_interp.remove_bias(prominence, 800)
        prominence = normalize_std(prominence)
        return prominence

    def _extract_word_duration(self, lab_lines):
        # assumes the breaks are already removed from lab_lines
        # print(f"Extracting word duration from {lab_lines}")
        word_duration = [float(end) - float(start) for start, end, _ in lab_lines]
        if self.word_duration_mode == "char_norm":
            word_duration = [
                duration / len(word)
                for duration, (_, _, word) in zip(word_duration, lab_lines)
            ]
        elif self.word_duration_mode == "absolute":
            # round to 3 decimals because this is the max precision of our alignment
            word_duration = [round(duration, 3) for duration in word_duration]
        elif self.word_duration_mode == "syllable_norm":
            word_duration = [
                duration / nb_syllables(word) if nb_syllables(word) > 0 else 0
                for duration, (_, _, word) in zip(word_duration, lab_lines)
            ]
        else:
            raise ValueError(f"Unknown word_duration_mode: {self.word_duration_mode}")
        return word_duration

    def _extract_feature_per_word(self, lab_lines, feature):
        end_time = float(lab_lines[-1][1])
        features = []
        for start, end, word in lab_lines:
            start_idx = sec_to_idx(float(start), end_time, len(feature))
            end_idx = sec_to_idx(float(end), end_time, len(feature))
            features.append(feature[start_idx:end_idx])
        return features

    def _extract_f0_per_word(self, lab_lines, f0, phoneme_lab_lines, verbose=False):
        """
        Extracts f0 per word from the f0 signal.
        Here we can make choices for the stress localization
        """
        cnt_not_found = 0
        end_time = float(lab_lines[-1][1])
        f0_per_word = []
        for start, end, word in lab_lines:
            start_idx = sec_to_idx(float(start), end_time, len(f0))
            end_idx = sec_to_idx(float(end), end_time, len(f0))
            if verbose:
                print(f"original start idx: {start_idx}, end idx: {end_idx}")

            if self.f0_stress_localizer == "celex":
                syllables = syllabify(word)
                stressed_syllable_idx = self.celex_manager.get_stress_index(word)

                stress_syllable_time = find_stress_syllable_start(
                    syllables=syllables,
                    stress_index=stressed_syllable_idx,
                    phoneme_lab_lines=phoneme_lab_lines,
                    word_start=float(start),
                    word_end=float(end),
                )

                if stress_syllable_time:
                    stress_syllable_idx = sec_to_idx(
                        stress_syllable_time, end_time, len(f0)
                    )
                    if verbose:
                        print(
                            f"stress syllable found for {word} at {stress_syllable_idx}"
                        )
                    new_start = max(
                        start_idx, stress_syllable_idx - self.f0_window // 2
                    )
                    new_end = min(end_idx, stress_syllable_idx + self.f0_window // 2)
                    if verbose:
                        print(f"new start idx: {new_start}, end idx: {new_end}")
                else:
                    if verbose:
                        print(f"no stress syllable found for {word}")
                    cnt_not_found += 1
                    new_start = start_idx
                    new_end = end_idx

                # resample
                f0_per_word.append(
                    signal.resample(f0[new_start:new_end], self.f0_resampling_length)
                )

            else:
                raise NotImplementedError(
                    f"Unknown f0_stress_localizer: {self.f0_stress_localizer}"
                )

        return f0_per_word, cnt_not_found

    def _extract_features(self, lab_path: str, wav_path: str, phoneme_lab_path: str):
        features = {}
        nb_syllables_not_found = 0

        if self.extract_f0 or self.extract_energy or self.extract_duration:
            fs, waveform = read_wav(wav_path)
        lab_lines = read_lab_file(lab_path)  # contains breaks
        phoneme_lab_lines = read_lab_file(phoneme_lab_path)  # contains breaks
        f0, energy, duration, prominence = None, None, None, None

        # need full lab lines for the pauses
        if self.extract_pause_before:
            pause_before = self.pause_before_extractor.extract_from_lab_lines(
                lines=lab_lines
            )  # format [(word, {"before": pause), ...]
            if pause_before is None:
                return None, None  # alignment failed
            pause_before = [
                round(pause, 3) for pause in pause_before if pause is not None
            ]  # maximum precision of MFA
            features["pause_before"] = pause_before
        if self.extract_pause_after:
            pause_after = self.pause_after_extractor.extract_from_lab_lines(
                lines=lab_lines
            )
            if pause_after is None:
                return None, None  # alignment failed
            pause_after = [
                round(pause, 3) for pause in pause_after if pause is not None
            ]  # maximum precision of MFA
            features["pause_after"] = pause_after

        # remove pauses from lab lines
        lab_lines = remove_breaks_from_lab_lines(lab_lines)
        phoneme_lab_lines = remove_breaks_from_lab_lines(phoneme_lab_lines)

        words = [word for _, _, word in lab_lines]
        # check for invalid words
        if any(
            [word in self.unallowed_symbols for word in words]
        ):  # O(words*unallowed_words)
            return None, None

        # check for unallowed words
        features["words"] = words

        # First extract the features over the audio file
        if self.extract_f0:
            f0 = self._extract_f0(waveform, fs)
        if self.extract_energy:
            energy = self._extract_energy(waveform, fs)
        if self.extract_duration:
            resample_length = min_length_of_lists([f0, energy])
            duration = self._extract_duration(waveform, fs, lab_lines, resample_length)
        if self.extract_prominence:
            prominence = self._extract_prominence(f0, energy, duration)

        if self.extract_word_duration:
            word_duration = self._extract_word_duration(lab_lines)
            features["word_duration"] = word_duration

        assert equal_length_or_none([f0, energy, duration, prominence])

        # Then extract the features over the words
        if self.extract_f0:
            f0_per_word, cnt_not_found = self._extract_f0_per_word(
                lab_lines, f0, phoneme_lab_lines
            )
            nb_syllables_not_found += cnt_not_found
            # print("f0 per word", f0_per_word)
            f0_per_word_parameterized = [self._parameterize_f0(f) for f in f0_per_word]
            features["f0_parameterized"] = f0_per_word_parameterized

        if self.extract_energy:
            energy_per_word = self._extract_feature_per_word(lab_lines, energy)
            if self.energy_mode == "mean":
                energy_per_word = [np.mean(e) for e in energy_per_word]
            elif self.energy_mode == "max":
                energy_per_word = [np.max(e) for e in energy_per_word]
            elif self.energy_mode == "curve":
                pass  # already extracted
            else:
                raise ValueError(
                    f"energy_mode {self.energy_mode} not supported. Use 'mean' or 'max'"
                )
            features["energy"] = energy_per_word

        if self.extract_duration:
            duration_per_word = self._extract_feature_per_word(lab_lines, duration)
            features["duration"] = duration_per_word

        if self.extract_prominence:
            prominence_per_word = self._extract_feature_per_word(lab_lines, prominence)
            if self.prominence_mode == "mean":
                prominence_per_word = [np.mean(p) for p in prominence_per_word]
            elif self.prominence_mode == "max":
                prominence_per_word = [np.max(p) for p in prominence_per_word]
            elif self.prominence_mode == "curve":
                pass  # already extracted
            features["prominence"] = prominence_per_word

        return features, nb_syllables_not_found

    def _process_reader(self, reader):
        if reader == ".DS_Store":
            return [], 0, 0

        reader_path = os.path.join(self.lab_root, reader)
        reader_samples = []
        failed_alignments = 0
        total_nb_syllables_not_found = 0

        for book in os.listdir(reader_path):
            if book == ".DS_Store":
                continue
            book_path = os.path.join(reader_path, book)
            book_files = [file for file in os.listdir(book_path) if file != ".DS_Store"]

            for ut in book_files:
                if ut == ".DS_Store":
                    continue
                ut_path_lab = os.path.join(book_path, ut)
                ut_path_wav = os.path.join(
                    self.wav_root, reader, book, ut.replace(".lab", ".wav")
                )
                ut_path_txt = os.path.join(
                    self.wav_root, reader, book, ut.replace(".lab", ".original.txt")
                )

                text = str(open(ut_path_txt).read())

                ut_path_phoneme_lab = os.path.join(
                    str(self.phoneme_lab_root),
                    str(ut_path_lab.replace(self.lab_root, "").lstrip("/")),
                )

                features, nb_syll_not_found = self._extract_features(
                    lab_path=ut_path_lab,
                    wav_path=ut_path_wav,
                    phoneme_lab_path=ut_path_phoneme_lab,
                )

                if features is None:
                    failed_alignments += 1
                    continue
                else:
                    total_nb_syllables_not_found += nb_syll_not_found

                sample = OrderedDict()
                sample["reader"] = reader
                sample["book"] = book
                sample["text"] = text
                sample["features"] = features
                sample["path_txt"] = ut_path_txt
                sample["path_lab"] = ut_path_lab
                sample["path_wav"] = ut_path_wav
                sample["filename"] = ut.replace(".lab", "")

                reader_samples.append(sample)

        return reader_samples, failed_alignments, total_nb_syllables_not_found

    def process_files(
        self,
        lab_root: str = None,
        wav_root: str = None,
        cpu_cores: int = 4,
        verbose=False,
    ):
        if lab_root:
            self.lab_root = lab_root
        if wav_root:
            self.wav_root = wav_root

        self.samples = []
        failed_alignments = 0
        total_nb_syllables_not_found = 0

        with cf.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
            readers = os.listdir(self.lab_root)
            future_to_reader = {
                executor.submit(self._process_reader, reader): reader
                for reader in readers
            }

            for future in cf.as_completed(future_to_reader):
                (
                    reader_samples,
                    nb_failed_alignments,
                    nb_syllables_not_found,
                ) = future.result()
                self.samples += reader_samples
                failed_alignments += nb_failed_alignments
                total_nb_syllables_not_found += nb_syllables_not_found

        if verbose:
            print("Failed alignments: ", failed_alignments)
            print("Total number of syllables not found: ", total_nb_syllables_not_found)
            print("Total aligned utterances: ", len(self.samples))

        return self.samples

    # def process_files(self, lab_root: str = None, wav_root: str = None, verbose=False):
    #     """
    #     Iterate over all files in the lab_root directory and extract the features.
    #     The features are stored in a dictionary with the following structure:
    #     {
    #         "reader": reader,
    #         "book": book,
    #         "text": text,
    #         "features": features, # this is again a dict which contains key value pairs of the feature names and the feature values
    #         "path_txt": ut_path_txt,
    #         "path_lab": ut_path_lab,
    #         "path_wav": ut_path_wav,
    #         }
    #     """
    #     if lab_root:
    #         self.lab_root = lab_root
    #     if wav_root:
    #         self.wav_root = wav_root

    #     self.samples = []
    #     failed_alignments = 0
    #     total_nb_syllables_not_found = 0

    #     for reader in tqdm(
    #         os.listdir(self.lab_root), desc="Extracting Features from Readers"
    #     ):
    #         if reader == ".DS_Store":
    #             continue

    #         reader_path = os.path.join(self.lab_root, reader)
    #         reader_samples = []

    #         for book in os.listdir(reader_path):
    #             if book == ".DS_Store":
    #                 continue
    #             book_path = os.path.join(reader_path, book)
    #             book_files = [
    #                 file for file in os.listdir(book_path) if file != ".DS_Store"
    #             ]

    #             for ut in book_files:
    #                 # print("utterance", ut)
    #                 if ut == ".DS_Store":
    #                     continue
    #                 ut_path_lab = os.path.join(book_path, ut)
    #                 # print("utterance lab path ", ut_path_lab)
    #                 ut_path_wav = os.path.join(
    #                     self.wav_root, reader, book, ut.replace(".lab", ".wav")
    #                 )
    #                 # print("utterance wav path ", ut_path_wav)
    #                 ut_path_txt = os.path.join(
    #                     self.wav_root, reader, book, ut.replace(".lab", ".original.txt")
    #                 )
    #                 if verbose:
    #                     print(f"Processing file {ut_path_lab}")

    #                 text = str(open(ut_path_txt).read())
    #                 if verbose:
    #                     print(f"Text: {text}")

    #                 # print(f"ut_lab_path: {ut_path_lab}")
    #                 # print(
    #                 #     f"replace {type(ut_path_lab.replace(self.lab_root, '').lstrip('/'))}"
    #                 # )
    #                 # create ut_path_phoneme_lab by removing self.lab_root from the front and adding self.phoneme_lab_root
    #                 ut_path_phoneme_lab = os.path.join(
    #                     str(self.phoneme_lab_root),
    #                     str(ut_path_lab.replace(self.lab_root, "").lstrip("/")),
    #                 )

    #                 features, nb_syll_not_found = self._extract_features(
    #                     lab_path=ut_path_lab,
    #                     wav_path=ut_path_wav,
    #                     phoneme_lab_path=ut_path_phoneme_lab,
    #                 )

    #                 if verbose:
    #                     print(f"Features: {features}")

    #                 if features is None:
    #                     failed_alignments += 1
    #                     continue
    #                 else:
    #                     total_nb_syllables_not_found += nb_syll_not_found

    #                 if verbose:
    #                     print(f"Extracted numer of features: {len(features)}")

    #                 sample = OrderedDict()
    #                 sample["reader"] = reader
    #                 sample["book"] = book
    #                 sample["text"] = text
    #                 sample["features"] = features
    #                 sample["path_txt"] = ut_path_txt
    #                 sample["path_lab"] = ut_path_lab
    #                 sample["path_wav"] = ut_path_wav
    #                 sample["filename"] = ut.replace(".lab", "")
    #                 # print(sample)
    #                 reader_samples.append(sample)
    #         # add the sentence to the list of all sentences
    #         self.samples += reader_samples

    #     if verbose:
    #         print("Failed alignments: ", failed_alignments)
    #         print("Total number of syllables not found: ", total_nb_syllables_not_found)
    #         print("Total aligned utterances: ", len(self.samples))

    #     return self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_text(self, idx):
        return self.samples[idx]["text"]

    def get_pause_before(self, idx):
        return self.samples[idx]["features"]["pause_before"]

    def get_pause_after(self, idx):
        return self.samples[idx]["features"]["pause_after"]

    def get_energy(self, idx):
        return self.samples[idx]["features"]["energy"]

    def get_word_duration(self, idx):
        return self.samples[idx]["features"]["word_duration"]

    def get_prominence(self, idx):
        return self.samples[idx]["features"]["prominence"]

    def get_f0(self, idx):
        return self.samples[idx]["features"]["f0_parameterized"]

    def get_filename(self, idx):
        return self.samples[idx]["filename"]

    def get_reader(self, idx):
        return self.samples[idx]["reader"]

    def get_book(self, idx):
        return self.samples[idx]["book"]

    def get_path_txt(self, idx):
        return self.samples[idx]["path_txt"]

    def get_path_lab(self, idx):
        return self.samples[idx]["path_lab"]

    def get_path_wav(self, idx):
        return self.samples[idx]["path_wav"]

    def get_features(self, idx):
        return self.samples[idx]["features"]

    def get_all_features(self):
        return [sample["features"] for sample in self.samples]

    def get_all_pause_before(self):
        return [sample["features"]["pause_before"] for sample in self.samples]

    def get_all_pause_after(self):
        return [sample["features"]["pause_after"] for sample in self.samples]

    def get_all_energy(self):
        return [sample["features"]["energy"] for sample in self.samples]

    def get_all_f0(self):
        return [sample["features"]["f0_parameterized"] for sample in self.samples]

    def get_all_word_duration(self):
        return [sample["features"]["word_duration"] for sample in self.samples]

    def get_all_prominence(self):
        return [sample["features"]["prominence"] for sample in self.samples]

    def get_all_text(self):
        return [sample["text"] for sample in self.samples]

    def get_all_filename(self):
        return [sample["filename"] for sample in self.samples]

    def get_all_reader(self):
        return [sample["reader"] for sample in self.samples]

    def get_all_book(self):
        return [sample["book"] for sample in self.samples]

    def get_all_path_txt(self):
        return [sample["path_txt"] for sample in self.samples]

    def get_all_path_lab(self):
        return [sample["path_lab"] for sample in self.samples]

    def get_all_path_wav(self):
        return [sample["path_wav"] for sample in self.samples]

    def get_all_features(self):
        return [sample["features"] for sample in self.samples]


class WordBreakFeatureExtractor:
    """
    Extract and access the features of the LibriTTS dataset.
    """

    def __init__(
        self,
        lab_root: str,
        wav_root: str,
        data_cache: str,
        lowercase: bool = False,
        remove_punctuation: bool = False,
    ):
        """
        samples: list of OrderedDict containing samples
        """
        self.feature_extractors = WordBreakExtractor(modes=["before", "after"])
        print(f"Searching for features from {lab_root}...")

        # check cache
        file_name = lab_root.split("/")[-1]
        cache_path = os.path.join(data_cache, file_name, "break_samples.pkl")
        if os.path.exists(cache_path):
            self.samples = pickle.load(open(cache_path, "rb"))
            print(f"Loaded {len(self.samples)} samples from cache.")
        else:
            print(f"No cache found. Extracting features {file_name} from scratch.")
            self.samples = get_duration_features_from_lab_root(
                lab_root, wav_root, self.feature_extractors
            )
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            pickle.dump(self.samples, open(cache_path, "wb"))
            print(f"Finished extracting {len(self.samples)} samples.")
            print(f"Saved samples to {cache_path}")

        self.length = len(self.samples)
        print(f"Finished extracting {self.length} samples.")
        self.filename_to_index = {self.get_filename(i): i for i in range(self.length)}

        # Preprocessing
        if lowercase and remove_punctuation:
            self.preprocess_fct = python_lowercase_remove_punctuation
        elif lowercase:
            self.preprocess_fct = python_lowercase
        elif remove_punctuation:
            self.preprocess_fct = python_remove_punctuation
        else:
            self.preprocess_fct = lambda x: x

        # Calculate stats
        self.all_breaks = self.get_all_breaks_before()
        self.mean_break = np.mean(np.concatenate(self.all_breaks))
        self.std_break = np.std(np.concatenate(self.all_breaks))

        # For word-specific statistics
        self.word_breaks = defaultdict(list)
        self.word_break_means = defaultdict(float)
        self.word_break_stds = defaultdict(float)

        for i in range(self.length):
            features = self.get_features(i)
            for word, features in features:
                self.word_breaks[word].append(features["before"])
        for word, breaks in self.word_breaks.items():
            self.word_break_means[word] = np.mean(breaks)
            self.word_break_stds[word] = np.std(breaks)

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return self.length

    def get_index(self, filename):
        return self.filename_to_index[filename]

    def get_filename(self, index):
        return self.samples[index]["filename"]

    def get_text(self, index):
        text = " ".join(self.samples[index]["text"])
        return self.preprocess_fct(text)

    def get_features(self, index):
        return self.samples[index]["features"]

    def get_break_before(self, index):
        # sample is a tuple (word, features)
        return [sample[1]["before"] for sample in self.samples[index]["features"]]

    def get_break_after(self, index):
        # sample is a tuple (word, features)
        return [sample[1]["after"] for sample in self.samples[index]["features"]]

    def get_all_breaks_before(self):
        return [self.get_break_before(i) for i in range(self.length)]

    def get_all_breaks_after(self):
        return [self.get_break_after(i) for i in range(self.length)]

    def get_all_texts(self):
        return [self.get_text(i) for i in range(self.length)]

    def get_all_features(self):
        return [self.get_features(i) for i in range(self.length)]

    def get_max_text_length_in_words(self):
        return max([len(self.get_text(i).split(" ")) for i in range(self.length)])

    def get_break_stats(self):
        return self.mean_break, self.std_break

    def get_word_stats(self, word):
        return self.word_break_means[word], self.word_break_stds[word]


class DurationFeatureExtractor:
    """
    Extract and access the features of the LibriTTS dataset.
    """

    def __init__(
        self,
        lab_root: str,
        wav_root: str,
        data_cache: str,
        lowercase: bool = False,
        remove_punctuation: bool = False,
    ):
        """
        samples: list of OrderedDict containing samples
        """
        self.feature_extractors = DurationExtractor(
            modes=["duration", "duration_scaled"]
        )
        print(f"Searching for features from {lab_root}...")

        # check cache
        file_name = lab_root.split("/")[-1]
        cache_path = os.path.join(data_cache, file_name, "duration_samples.pkl")
        if os.path.exists(cache_path):
            self.samples = pickle.load(open(cache_path, "rb"))
            print(f"Loaded {len(self.samples)} samples from cache.")
        else:
            print(f"No cache found. Extracting features {file_name} from scratch.")
            self.samples = get_duration_features_from_lab_root(
                lab_root, wav_root, self.feature_extractors
            )
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            pickle.dump(self.samples, open(cache_path, "wb"))
            print(f"Finished extracting {len(self.samples)} samples.")
            print(f"Saved samples to {cache_path}")

        self.length = len(self.samples)
        print(f"Finished extracting {self.length} samples.")
        self.filename_to_index = {self.get_filename(i): i for i in range(self.length)}

        # Preprocessing
        if lowercase and remove_punctuation:
            self.preprocess_fct = python_lowercase_remove_punctuation
        elif lowercase:
            self.preprocess_fct = python_lowercase
        elif remove_punctuation:
            self.preprocess_fct = python_remove_punctuation
        else:
            self.preprocess_fct = lambda x: x

        # Calculate stats
        self.all_durations = self.get_all_durations()
        self.mean_duration = np.mean(np.concatenate(self.all_durations))
        self.std_duration = np.std(np.concatenate(self.all_durations))

        # For word-specific statistics
        self.word_durations = defaultdict(list)
        self.word_durations_scaled = defaultdict(list)
        self.word_means = defaultdict(float)
        self.word_stds = defaultdict(float)
        for i in range(self.length):
            features = self.get_features(i)
            for word, features in features:
                self.word_durations[word].append(features["duration"])
                self.word_durations_scaled[word].append(features["duration_scaled"])
        for word, durations in self.word_durations.items():
            self.word_means[word] = np.mean(durations)
            self.word_stds[word] = np.std(durations)

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return self.length

    def get_index(self, filename):
        return self.filename_to_index[filename]

    def get_filename(self, index):
        return self.samples[index]["filename"]

    def get_text(self, index):
        text = " ".join(self.samples[index]["text"])
        return self.preprocess_fct(text)

    def get_features(self, index):
        return self.samples[index]["features"]

    def get_duration(self, index):
        # sample is a tuple (word, features)
        return [sample[1]["duration"] for sample in self.samples[index]["features"]]

    def get_scaled_duration(self, index):
        return [
            sample[1]["duration_scaled"] for sample in self.samples[index]["features"]
        ]

    def get_all_durations(self):
        return [self.get_duration(i) for i in range(self.length)]

    def get_all_scaled_durations(self):
        return [self.get_scaled_duration(i) for i in range(self.length)]

    def get_all_texts(self):
        return [self.get_text(i) for i in range(self.length)]

    def get_all_features(self):
        return [self.get_features(i) for i in range(self.length)]

    def get_max_text_length_in_words(self):
        return max([len(self.get_text(i).split(" ")) for i in range(self.length)])

    def get_duration_stats(self):
        return self.mean_duration, self.std_duration

    def get_word_stats(self, word):
        return self.word_means[word], self.word_stds[word]


class LibriTTSFeatureExtractor:
    """
    Extract and access the features of the LibriTTS dataset.
    """

    def __init__(
        self,
        lab_root,
        wav_root,
        data_cache="./",
        # feature_extractors,
        lowercase=False,
        remove_punctuation=False,
    ):
        """
        samples: list of OrderedDict containing samples
        """
        self.feature_extractors = F0Extractor(
            modes=["min", "max", "mean", "curve"],
            interpolate=False,
            keep_unvoiced=False,
        )
        print(f"Searching for features from {lab_root}...")
        # check cache
        file_name = lab_root.split("/")[-1]
        cache_path = os.path.join(data_cache, file_name, "f0_samples.pkl")
        if os.path.exists(cache_path):
            self.samples = pickle.load(open(cache_path, "rb"))
            print(f"Loaded {len(self.samples)} samples from cache.")
        else:
            print(f"No cache found. Extracting features {file_name} from scratch.")
            self.samples = get_features_from_lab_root(
                lab_root, wav_root, self.feature_extractors
            )
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            pickle.dump(self.samples, open(cache_path, "wb"))
            print(f"Finished extracting {len(self.samples)} samples.")
            print(f"Saved samples to {cache_path}")

        self.length = len(self.samples)
        print(f"Finished extracting {self.length} samples.")
        self.filename_to_index = {self.get_filename(i): i for i in range(self.length)}

        # Preprocessing
        if lowercase and remove_punctuation:
            self.preprocess_fct = python_lowercase_remove_punctuation
        elif lowercase:
            self.preprocess_fct = python_lowercase
        elif remove_punctuation:
            self.preprocess_fct = python_remove_punctuation
        else:
            self.preprocess_fct = lambda x: x

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return self.length

    def get_index(self, filename):
        return self.filename_to_index[filename]

    def get_filename(self, index):
        return self.samples[index]["filename"]

    def get_text(self, index):
        text = " ".join(self.samples[index]["text"])
        return self.preprocess_fct(text)

    def get_features(self, index):
        return self.samples[index]["features"]

    def get_f0_mean(self, index):
        return [sample[1]["mean"] for sample in self.samples[index]["features"]]

    def get_f0_min(self, index):
        return [sample[1]["min"] for sample in self.samples[index]["features"]]

    def get_f0_max(self, index):
        return [sample[1]["max"] for sample in self.samples[index]["features"]]

    def get_f0_curve(self, index):
        return [sample[1]["curve"] for sample in self.samples[index]["features"]]

    def get_all_f0_mean(self):
        return [self.get_f0_mean(i) for i in range(self.length)]

    def get_all_f0_curve(self):
        return [self.get_f0_curve(i) for i in range(self.length)]

    def get_all_texts(self):
        return [self.get_text(i) for i in range(self.length)]

    def get_all_features(self):
        return [self.get_features(i) for i in range(self.length)]

    def get_max_text_length_in_words(self):
        return max([len(self.get_text(i).split(" ")) for i in range(self.length)])
