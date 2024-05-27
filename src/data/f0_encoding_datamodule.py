from argparse import ArgumentError
from typing import Optional, Tuple
import os
import json
import pickle

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


# from src.data.components.helsinki import HelsinkiProminenceExtractor
from src.data.components.feature_extractors import LibriTTSFeatureExtractor
from src.data.components.datasets import TimeSeriesDataset
from src.data.components.collators import rnn_collate_fn


class F0EncodingDataModule(LightningDataModule):
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
        train_file: str,
        val_file: str,
        test_file: str,
        file_storage: str,
        dataset_name: str,
        # use_fast_tokenizer: bool = False,
        batch_size: int = 64,
        max_length: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        model_name: str = None,
        # score_first_token: bool = False,
        # relative_to_prev: bool = False,
        # n_prev: int = 1,
        # relative_to_mean: bool = False,
        # word_stats_path: str = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = None
        # self.tokenizer = None
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

    def prepare_dataset(self, file_key):
        if os.path.exists(
            os.path.join(
                self.hparams.file_storage,
                f"f0_texts-{getattr(self.hparams, file_key)}.pkl",
            )
        ) and os.path.exists(
            os.path.join(
                self.hparams.file_storage,
                f"f0_curves-{getattr(self.hparams, file_key)}.pkl",
            )
        ):
            print(
                f"Loading pickled texts and f0 labels from {self.hparams.file_storage}"
            )
            texts = pickle.load(
                open(
                    os.path.join(
                        self.hparams.file_storage,
                        f"f0_texts-{getattr(self.hparams, file_key)}.pkl",
                    ),
                    "rb",
                )
            )
            f0_curves = pickle.load(
                open(
                    os.path.join(
                        self.hparams.file_storage,
                        f"f0_curves-{getattr(self.hparams, file_key)}.pkl",
                    ),
                    "rb",
                )
            )
        else:
            print(f"Extracting {file_key} texts and f0 labels from LibriTTS")
            extractor = LibriTTSFeatureExtractor(
                os.path.join(self.hparams.lab_root, getattr(self.hparams, file_key)),
                os.path.join(self.hparams.wav_root, getattr(self.hparams, file_key)),
            )

            texts = extractor.get_all_texts()
            f0_curves = extractor.get_all_f0_curve()

            # pickle the texts and f0 labels
            print(f"Storing pickled texts and f0 labels in {self.hparams.file_storage}")
            with open(
                os.path.join(
                    self.hparams.file_storage,
                    f"f0_texts-{getattr(self.hparams, file_key)}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(texts, f)
            with open(
                os.path.join(
                    self.hparams.file_storage,
                    f"f0_curves-{getattr(self.hparams, file_key)}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(f0_curves, f)

        # flatten out the curves (all words instead of sentences)
        words = [word for text in texts for word in text]
        f0_curves = [c for curve in f0_curves for c in curve]

        return texts, f0_curves, words

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        self.train_texts, self.train_f0_curves, self.train_words = self.prepare_dataset(
            "train_file"
        )
        self.val_texts, self.val_f0_curves, self.val_words = self.prepare_dataset(
            "val_file"
        )
        self.test_texts, self.test_f0_curves, self.test_words = self.prepare_dataset(
            "test_file"
        )

        self.train_dataset = TimeSeriesDataset(self.train_f0_curves)
        self.val_dataset = TimeSeriesDataset(self.val_f0_curves)
        self.test_dataset = TimeSeriesDataset(self.test_f0_curves)

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def collate(self, batch):
        return rnn_collate_fn(batch)

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
