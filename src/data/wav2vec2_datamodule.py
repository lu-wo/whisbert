from typing import Optional, Tuple
import pickle
import os, sys
import json


# import datasets
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
import pandas as pd

from transformers import Wav2Vec2FeatureExtractor
from src.utils.audio_utils import get_flac_files
from src.data.components.datasets import AudioDataset


class Wav2Vec2DataModule(LightningDataModule):
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
        sample_rate: int = 16000,
        min_seq_len: int = 9,
        max_seq_len: int = 15,
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
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base"
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
                self.flac_files = df["audio"].tolist()
            elif self.hparams.audio_root:
                print(f"Loading flac files from audio root {self.hparams.audio_root}")
                self.flac_files = get_flac_files(self.hparams.audio_root)
            else:
                raise ValueError(
                    "Either `file_mapping_path` or `audio_root` must be provided."
                )
            print(f"Found {len(self.flac_files)} flac files.")

        # Calculate split sizes
        train_size = self.hparams.train_val_test_split[0]
        val_size = self.hparams.train_val_test_split[1]
        test_size = self.hparams.train_val_test_split[2]

        # Split the data
        train_files, val_test_files = train_test_split(
            self.flac_files,
            test_size=val_size + test_size,
            random_state=self.hparams.seed,
        )
        val_files, test_files = train_test_split(
            val_test_files,
            test_size=test_size / (val_size + test_size),
            random_state=self.hparams.seed,
        )

        # Helper function to create dataset
        def create_dataset(files):
            return AudioDataset(
                flac_files=files,
                sr=self.hparams.sample_rate,
                min_seq_len=self.hparams.min_seq_len,
                max_seq_len=self.hparams.max_seq_len,
            )

        # Create datasets
        self.train_dataset = create_dataset(train_files)
        self.val_dataset = create_dataset(val_files)
        self.test_dataset = create_dataset(test_files)

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def collate(self, batch):
        """
        Collate function for DataLoader.
        ::batch: list of numpy arrays of audio signals
        """
        # Use the feature_extractor to process the batch of audio signals
        inputs = self.feature_extractor(
            batch,
            sampling_rate=self.hparams.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        return inputs

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
