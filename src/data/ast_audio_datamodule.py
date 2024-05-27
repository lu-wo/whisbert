from typing import Optional, Tuple
import pickle
import os, sys
import json


# import datasets
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import AutoTokenizer, BertTokenizer
from torch.utils.data import random_split
import torch

from src.data.components.feature_extractors import ProsodyFeatureExtractor
from src.data.components.datasets import StringDataset
from src.data.components.collators import collate_text
from src.utils.text_processing import TextFileProcessor
from src.data.components.datasets import ASTAudioDataset


class ASTAudioDataModule(LightningDataModule):
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
        data_root: str = None,  # peoples speech
        min_confidence: float = 0.0,
        train_file: str = None,  # libritts train file
        val_file: str = None,
        test_file: str = None,
        lab_root: str = None,
        wav_root: str = None,
        phoneme_lab_root: str = None,
        tokenizer_path: str = None,
        use_fast_tokenizer: bool = False,
        batch_size: int = 64,
        max_length: int = 512,
        num_workers: int = 4,
        pin_memory: bool = False,
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        model_name: str = None,
        debug: bool = False,
        seed: int = 0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = None
        self.collator_fn = None

        if tokenizer_path:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")
        self.pad_token_id = self.tokenizer.pad_token_id
        print(f"Dataloader: padding with token id: {self.pad_token_id}")

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

    def _prepare_ps_dataset(self):
        processor = TextFileProcessor(
            self.hparams.data_root, self.hparams.min_confidence
        )
        samples = processor.create_text_samples()
        num_words = processor.count_words_in_samples(samples)
        print(f"Number of words: {num_words}")

        # Split into train, val, test
        train_samples, val_samples, test_samples = processor.split_samples(
            samples=samples, seed=self.hparams.seed
        )

        # Create datasets
        train_dataset = StringDataset(list_of_strings=train_samples)
        val_dataset = StringDataset(list_of_strings=val_samples)
        test_dataset = StringDataset(list_of_strings=test_samples)

        return train_dataset, val_dataset, test_dataset

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""

        dataset = ASTAudioDataset(
            self.data_root,
            self.sr,
            self.min_seq_len,
            self.max_seq_len,
            self.max_length,
            self.padding_value,
            self.num_mel_bins,
            self.return_attention_mask,
        )

        num_train = int(self.train_val_test_split[0] * len(dataset))
        num_val = int(self.train_val_test_split[1] * len(dataset))

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset,
            [num_train, num_val, len(dataset) - num_train - num_val],
            generator=torch.Generator().manual_seed(self.seed),
        )
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

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
