import os
from tokenizers import (
    ByteLevelBPETokenizer,
    Tokenizer,
    models,
    trainers,
)
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import (
    BpeTrainer,
    WordLevelTrainer,
    WordPieceTrainer,
    UnigramTrainer,
)
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer

from src.utils.text_processing import TextFileProcessor


class TokenizerTrainer:
    def __init__(self, data, vocab_size, special_tokens_map):
        self.data = data
        self.vocab_size = vocab_size
        self.special_tokens_map = special_tokens_map
        self.tokenizer = None

    def train(self):
        # batch the texts into chunks of 1000
        # this is to avoid memory issues
        # when training the tokenizer
        batch_size = 1000
        batches = [
            self.data[i : i + batch_size] for i in range(0, len(self.data), batch_size)
        ]

        # iterator over the texts
        def text_iterator():
            for batch in batches:
                yield batch

        # load old tokenizer
        old_tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")

        self.tokenizer = old_tokenizer.train_new_from_iterator(
            text_iterator(),
            vocab_size=self.vocab_size,
            special_tokens_map=self.special_tokens_map,
        )

    def save(self, path):
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)


if __name__ == "__main__":
    # train on people's speech:
    root_dir = "/Users/lukas/Desktop/Projects/MIT/MIT_prosody/data/peoples-speech-clean"
    processor = TextFileProcessor(root_dir)
    data = processor.create_text_samples()
    num_samples = processor.count_words_in_samples(data)
    print(f"Number of words: {num_samples}")

    target_tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")

    trainer = TokenizerTrainer(
        data, target_tokenizer.vocab_size, target_tokenizer.special_tokens_map
    )
    trainer.train()
    trainer.save("./tokenizer_ps_10M")
