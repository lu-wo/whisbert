from typing import Optional, Tuple
import pickle
import os, sys
import json


# import datasets
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import AutoTokenizer, BertTokenizer

from src.data.components.feature_extractors import ProsodyFeatureExtractor
from src.data.components.datasets import StringDataset
from src.data.components.collators import collate_text
from src.utils.text_processing import TextFileProcessor


class FlavaTextDataModule(LightningDataModule):
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
        max_words: int = 80,
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
        num_workers: int = 0,
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

    def _prepare_libritts_dataset(self, file_name):
        data_cache_path = os.path.join(self.hparams.data_cache, file_name)
        data_id = f"flava_text_{self.hparams.dataset_name}.pkl"

        if os.path.exists(data_cache_path) and data_id in os.listdir(data_cache_path):
            print(f"Loading data from cache: {data_cache_path, data_id}")
            # Load the data from cache
            with open(os.path.join(data_cache_path, data_id), "rb") as f:
                data = pickle.load(f)
            texts = data["texts"]
        else:
            print(f"Data not in cache: {data_cache_path, data_id}")
            # Data not in cache, create it
            extractor = ProsodyFeatureExtractor(
                lab_root=os.path.join(self.hparams.lab_root, file_name),
                wav_root=os.path.join(self.hparams.wav_root, file_name),
                phoneme_lab_root=os.path.join(self.hparams.phoneme_lab_root, file_name),
                data_cache=self.hparams.data_cache,
            )
            texts = extractor.get_all_text()

            # Save the data to cache
            data = {"texts": texts}
            if not os.path.exists(data_cache_path):
                os.makedirs(data_cache_path)
            with open(os.path.join(data_cache_path, data_id), "wb") as f:
                pickle.dump(data, f)

            print(f"Saved data to cache: {data_cache_path, data_id}")

        print(f"Loaded {len(texts)} many strings")
        dataset = StringDataset(list_of_strings=texts)

        return texts, dataset

    def _prepare_ps_dataset(self):
        data_cache_path = os.path.join(
            self.hparams.data_cache,
            f"ps_dataset{self.hparams.max_words}_{self.hparams.min_confidence}",
        )
        data_id = f"flava_text_{self.hparams.dataset_name}.pkl"

        if os.path.exists(data_cache_path) and data_id in os.listdir(data_cache_path):
            print(f"Loading data from cache: {data_cache_path, data_id}")
            # Load the data from cache
            with open(os.path.join(data_cache_path, data_id), "rb") as f:
                data = pickle.load(f)
            train_samples = data["train_samples"]
            val_samples = data["val_samples"]
            test_samples = data["test_samples"]
        else:
            print(f"Data not in cache: {data_cache_path, data_id}")
            # Data not in cache, create it
            processor = TextFileProcessor(
                root_dirs=self.hparams.data_root,
                max_words=self.hparams.max_words,
                min_confidence=self.hparams.min_confidence,
            )
            samples = processor.create_text_samples()
            num_words = processor.count_words_in_samples(samples)
            print(f"Number of words: {num_words}")

            # Split into train, val, test
            train_samples, val_samples, test_samples = processor.split_samples(
                samples=samples, seed=self.hparams.seed
            )

            # Save the data to cache
            data = {
                "train_samples": train_samples,
                "val_samples": val_samples,
                "test_samples": test_samples,
            }
            if not os.path.exists(data_cache_path):
                os.makedirs(data_cache_path)
            with open(os.path.join(data_cache_path, data_id), "wb") as f:
                pickle.dump(data, f)

            print(f"Saved data to cache: {data_cache_path, data_id}")

        # Create datasets
        train_dataset = StringDataset(list_of_strings=train_samples)
        val_dataset = StringDataset(list_of_strings=val_samples)
        test_dataset = StringDataset(list_of_strings=test_samples)

        return train_dataset, val_dataset, test_dataset

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        if self.hparams.dataset_name == "libritts":
            (
                self.train_texts,
                self.train_dataset,
            ) = self._prepare_libritts_dataset(self.hparams.train_file)
            self.val_texts, self.val_dataset = self._prepare_libritts_dataset(
                self.hparams.val_file
            )
            self.test_texts, self.test_dataset = self.prepare_libritts_dataset(
                self.hparams.test_file
            )
        elif self.hparams.dataset_name == "peoples_speech":
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = self._prepare_ps_dataset()
        else:
            raise ValueError(f"Dataset name {self.hparams.dataset_name} not supported.")

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def collate(self, batch):
        return collate_text(batch, self.tokenizer)

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
