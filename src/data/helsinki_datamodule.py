from argparse import ArgumentError
from typing import Optional, Tuple

import os
import torch
import datasets
from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import DataCollatorWithPadding

from src.data.components.helsinki import HelsinkiProminenceExtractor
from src.data.components.datasets import TokenTaggingDataset


class HelsinkiDataModule(LightningDataModule):
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
        data_dir: str,
        train_file: str,
        val_file: str,
        test_file: str,
        dataset_name: str,
        tokenizer_name: str,
        batch_size: int = 64,
        max_length: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        print(f"DataModule: \n{self.hparams}")

        self.dataset = None
        self.tokenizer = None
        self.collator_fn = None

        # self.eval_key = "validation"
        # self.test_key = "test"

        self.keep_columns = [
            "idx",
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
            "bias",
            "teacher_probs",
        ]

    # @property
    # def num_classes(self) -> int:
    #     return 3

    def prepare_data(self):
        """
        We should not assign anything here, so this function simply ensures
        that the pre-processed data is available.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        if not self.tokenizer:
            # TODO: Load according to model-name
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name, use_fast=True
            )

        self.dataset_path = Path(self.hparams.data_dir)
        if not os.path.exists(self.dataset_path):
            raise ValueError("The provided folder does not exist.")

        train_extractor = HelsinkiProminenceExtractor(
            self.dataset_path, self.hparams.train_file
        )
        self.train_texts = train_extractor.get_all_texts()
        self.train_prominences = train_extractor.get_all_real_prominence()
        self.train_dataset = TokenTaggingDataset(
            self.train_texts, self.train_prominences, self.tokenizer
        )

        val_extractor = HelsinkiProminenceExtractor(
            self.dataset_path, self.hparams.val_file
        )
        self.val_texts = val_extractor.get_all_texts()
        self.val_prominences = val_extractor.get_all_real_prominence()
        self.val_dataset = TokenTaggingDataset(
            self.val_texts, self.val_prominences, self.tokenizer
        )

        test_extractor = HelsinkiProminenceExtractor(
            self.dataset_path, self.hparams.test_file
        )
        self.test_texts = test_extractor.get_all_texts()
        self.test_prominences = test_extractor.get_all_real_prominence()
        self.test_dataset = TokenTaggingDataset(
            self.test_texts, self.test_prominences, self.tokenizer
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
            collate_fn=self.collator_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )
