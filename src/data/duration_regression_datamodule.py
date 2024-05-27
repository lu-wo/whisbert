from argparse import ArgumentError
from typing import Optional, Tuple
import pickle
import os, sys
import json

import torch

# import datasets
from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import DataCollatorWithPadding
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer, AutoModel

from src.data.components.feature_extractors import ProsodyFeatureExtractor
from src.data.components.datasets import TokenTaggingDataset
from src.data.components.collators import collate_fn, encode_and_pad_batch


class DurationRegressionDataModule(LightningDataModule):
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
        lab_root: str,
        wav_root: str,
        phoneme_lab_root: str,
        data_cache: str,
        train_file: str,
        val_file: str,
        test_file: str,
        dataset_name: str,
        use_fast_tokenizer: bool = False,
        batch_size: int = 64,
        max_length: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        model_name: str = None,
        word_duration_mode: str = "syllable_norm",
        score_first_token: bool = False,
        score_last_token: bool = False,
        relative_to_prev: bool = False,
        n_prev: int = 1,
        relative_to_mean: bool = False,
        word_stats_path: str = None,
        debug: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = None
        self.tokenizer = None
        self.collator_fn = None

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

    def prepare_dataset(self, file_name):
        data_cache_path = os.path.join(self.hparams.data_cache, file_name)
        data_id = f"duration_{self.hparams.word_duration_mode}"

        if os.path.exists(data_cache_path) and data_id in os.listdir(data_cache_path):
            print(f"Loading data from cache: {data_cache_path, data_id}")
            # Load the data from cache
            with open(os.path.join(data_cache_path, data_id), "rb") as f:
                data = pickle.load(f)
            texts, durations = (
                data["texts"],
                data["durations"],
            )
        else:
            # Data not in cache, create it
            extractor = ProsodyFeatureExtractor(
                lab_root=os.path.join(self.hparams.lab_root, file_name),
                wav_root=os.path.join(self.hparams.wav_root, file_name),
                phoneme_lab_root=os.path.join(self.hparams.phoneme_lab_root, file_name),
                data_cache=self.hparams.data_cache,
                extract_word_duration=True,
                word_duration_mode=self.hparams.word_duration_mode,
            )
            texts = extractor.get_all_text()
            durations = extractor.get_all_word_duration()

            # Save the data to cache
            data = {"texts": texts, "durations": durations}
            if not os.path.exists(data_cache_path):
                os.makedirs(data_cache_path)
            with open(os.path.join(data_cache_path, data_id), "wb") as f:
                pickle.dump(data, f)

            print(f"Saved data to cache: {data_cache_path, data_id}")

        dataset = TokenTaggingDataset(
            input_texts=texts,
            targets=durations,
            tokenizer=self.tokenizer,
            model_name=self.hparams.model_name,
            score_first_token=self.hparams.score_first_token,
            score_last_token=self.hparams.score_last_token,
            relative_to_prev=self.hparams.relative_to_prev,
            n_prev=self.hparams.n_prev,
            relative_to_mean=self.hparams.relative_to_mean,
            word_stats=self.hparams.word_stats,
            debug=self.hparams.debug,
        )

        return texts, durations, dataset

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        if not self.hparams.relative_to_mean:
            self.hparams.word_stats = None  # pass None to the dataset
        elif self.hparams.relative_to_mean and not self.hparams.word_stats_path:
            raise ValueError(
                "If relative_to_mean is True, you must provide a word_stats_path."
            )
        else:
            self.hparams.word_stats = json.load(open(self.hparams.word_stats_path, "r"))

        if not self.tokenizer:
            if "gpt2" in self.hparams.model_name:
                print("Using GPT2 tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.hparams.model_name, add_prefix_space=True
                )
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            elif "bert" in self.hparams.model_name.lower():
                print(f"Using {self.hparams.model_name} tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
                self.tokenizer.pad_token_id = self.tokenizer.sep_token_id
            else:
                raise ValueError("Model name not recognized.")
        self.pad_token_id = self.tokenizer.pad_token_id
        print(f"Dataloader: padding with token id: {self.pad_token_id}")

        (
            self.train_texts,
            self.train_durations,
            self.train_dataset,
        ) = self.prepare_dataset(self.hparams.train_file)
        self.val_texts, self.val_durations, self.val_dataset = self.prepare_dataset(
            self.hparams.val_file
        )
        self.test_texts, self.test_durations, self.test_dataset = self.prepare_dataset(
            self.hparams.test_file
        )

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def encode_collate(self, batch):
        return encode_and_pad_batch(batch, self.tokenizer, self.hparams.model_name)

    def collate(self, batch):
        return collate_fn(batch, self.tokenizer.pad_token_id)

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
