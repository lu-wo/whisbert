from typing import Optional, Tuple
import pickle
import os, sys
import json
import random


# import datasets
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
import pandas as pd

from transformers import WhisperFeatureExtractor, AutoTokenizer
from src.utils.audio_utils import get_flac_files
from src.data.components.datasets import PSPackagedDataset, PeoplesMultiModalDataset


class WhisBertDataModule(LightningDataModule):
    """
    LightningDataModule for HF Datasets.
    Requires a pre-processed (tokenized, cleaned...) dataset provided within the `data` folder.
    Might require adjustments if your dataset doesn't follow the structure of SNLI or MNLI.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_cache: str,
        dataset_name: str,
        model_name: str,
        text_root: str = None,  # peoples speech
        audio_root: str = None,  # peoples speech
        file_mapping_path: str = None,  # peoples speech
        mel_channels: int = 80,
        sample_rate: int = 16000,
        max_sentence_length: int = 50,
        dataset_total_words: int = 100e6,
        unmatch_samples: bool = False,
        padding_value: float = 0.0,
        chunk_length: int = 100,
        tokenizer_path: str = None,
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = False,
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        debug: bool = False,
        seed: int = 0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.collator_fn = None

        self.flac_files = None  # list of flac files, used in setup
        self.feature_extractor = WhisperFeatureExtractor(
            feature_size=mel_channels,
            sampling_rate=sample_rate,
            padding_value=padding_value,
            return_attention_mask=True,
            chunk_length=chunk_length,
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer_path)
            if tokenizer_path
            else AutoTokenizer.from_pretrained(model_name)
        )

        self.keep_columns = [
            "idx",
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
            "bias",
            "teacher_probs",
        ]

    def prepare_data(self):
        """
        We should not assign anything here, so this function simply ensures
        that the pre-processed data is available.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""

        print(f"Setting up {self.hparams.dataset_name} dataset.")

        if not self.flac_files:
            if self.hparams.file_mapping_path:
                print(f"Loading flac files from csv {self.hparams.file_mapping_path}")
                # load the "audio" column as strings from the csv stored at path
                df = pd.read_csv(self.hparams.file_mapping_path)
                # get the list of flac files
                self.text_files = df.iloc[:, 0].tolist()  # First column
                self.flac_files = df.iloc[:, 1].tolist()  # Second column
            elif self.hparams.audio_root and self.hparams.text_root:
                raise NotImplementedError(
                    "Loading flac files from audio and text root is not implemented yet."
                )
            else:
                raise ValueError(
                    "Either `file_mapping_path` or `audio_root` must be provided."
                )
            print(f"Found {len(self.flac_files)} flac files.")

        if self.hparams.unmatch_samples:
            print("Unmatching samples.")
            # permute the flac files such that they don't match the text files
            random.shuffle(self.flac_files)

        # Split audio and text files jointly
        train_flac, test_flac, train_texts, test_texts = train_test_split(
            self.flac_files,
            self.text_files,
            test_size=self.hparams.train_val_test_split[1]
            + self.hparams.train_val_test_split[2],
            random_state=self.hparams.seed,
        )

        val_flac, test_flac, val_texts, test_texts = train_test_split(
            test_flac,
            test_texts,
            test_size=self.hparams.train_val_test_split[2]
            / (
                self.hparams.train_val_test_split[1]
                + self.hparams.train_val_test_split[2]
            ),
            random_state=self.hparams.seed,
        )

        print(
            f"Number of train val test files: {len(train_flac)} {len(val_flac)} {len(test_flac)}"
        )
        train_fraction = int(
            self.hparams.train_val_test_split[0] * self.hparams.dataset_total_words
        )
        val_fraction = int(
            self.hparams.train_val_test_split[1] * self.hparams.dataset_total_words
        )
        test_fraction = int(
            self.hparams.train_val_test_split[2] * self.hparams.dataset_total_words
        )
        print(
            f"Number of train val test samples: {train_fraction} {val_fraction} {test_fraction}"
        )

        self.train_dataset = PeoplesMultiModalDataset(
            flac_files=train_flac,
            alignment_files=train_texts,
            sr=self.hparams.sample_rate,
            num_words_per_sample=self.hparams.max_sentence_length,
            dataset_total_words=train_fraction,
            unmatched_samples=self.hparams.unmatch_samples,
        )
        self.val_dataset = PeoplesMultiModalDataset(
            flac_files=val_flac,
            alignment_files=val_texts,
            sr=self.hparams.sample_rate,
            num_words_per_sample=self.hparams.max_sentence_length,
            dataset_total_words=val_fraction,
            unmatched_samples=self.hparams.unmatch_samples,
        )
        self.test_dataset = PeoplesMultiModalDataset(
            flac_files=test_flac,
            alignment_files=test_texts,
            sr=self.hparams.sample_rate,
            num_words_per_sample=self.hparams.max_sentence_length,
            dataset_total_words=test_fraction,
            unmatched_samples=self.hparams.unmatch_samples,
        )

        #
        # PACKS
        #
        # pack path
        # path_path = (
        #     "/Users/lukas/Desktop/Projects/MIT/data/peoples_speech/packs/100M_local_5"
        # )
        # # load all absolute filepaths
        # all_files = os.listdir(path_path)
        # all_files = [os.path.join(path_path, f) for f in all_files]
        # # TODO: remove this
        # # all_files = all_files[: int(len(all_files) * )]
        # # split into train, val, test
        # train_paths, test_paths = train_test_split(
        #     all_files, test_size=0.4, random_state=0
        # )
        # val_paths, test_paths = train_test_split(
        #     test_paths, test_size=0.5, random_state=0
        # )

        # # Helper function to create dataset
        # def create_dataset(pack_filepaths, num_files_per_pack=1000):
        #     return PSPackagedDataset(
        #         pack_filepaths=pack_filepaths,
        #         num_files_per_pack=num_files_per_pack,
        #         unmatch_samples=self.hparams.unmatch_samples,
        #     )
        # Create datasets
        # self.train_dataset = create_dataset(train_paths)
        # self.val_dataset = create_dataset(val_paths)
        # self.test_dataset = create_dataset(test_paths)

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def collate(self, batch):
        """
        Collate function for DataLoader.
        ::batch: list of numpy arrays of audio signals
        """
        # Use the feature_extractor to process the batch of audio signals
        audio_signals, texts = zip(*batch)
        # print(f"Audio signals: {audio_signals}")
        # print(f"Texts: {texts}")
        audio_inputs = self.feature_extractor(
            audio_signals,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest",
            sampling_rate=self.hparams.sample_rate,
        )

        # restrict tokenizer to 512 tokens
        text_inputs = self.tokenizer(
            texts,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
            max_length=128,
            truncation=True,
        )

        # print(f"Audio inputs: {audio_inputs.input_features.shape}")
        # print(f"Text inputs: {text_inputs.input_ids.shape}")

        return {
            "audio_inputs": audio_inputs,
            "text_inputs": text_inputs,
        }

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=False,
        )
