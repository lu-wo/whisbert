import os
import numpy as np
import torch
from tqdm import tqdm
from src.utils.text_processing import python_lowercase_remove_punctuation


class GloVeModel:
    def __init__(self, model_path):
        self.embeddings, self.word_to_idx = self.load_vectors(model_path)
        self.idx_to_word = {i: word for i, word in enumerate(self.word_to_idx)}
        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(self.embeddings).float()
        )

    @staticmethod
    def load_vectors(model_path):
        vocab, embeddings = [], []
        with open(model_path, "rt") as fi:
            full_content = fi.read().strip().split("\n")
        for i in tqdm(range(len(full_content)), desc="Loading GloVe"):
            i_word = full_content[i].split(" ")[0]
            i_embeddings = [float(val) for val in full_content[i].split(" ")[1:]]
            vocab.append(i_word)
            embeddings.append(i_embeddings)
        vocab_npa = np.array(vocab)
        embs_npa = np.array(embeddings)
        vocab_npa = np.insert(vocab_npa, 0, "<pad>")
        vocab_npa = np.insert(vocab_npa, 1, "<unk>")
        pad_emb_npa = np.zeros((1, embs_npa.shape[1]))  # embedding for '<pad>' token.
        unk_emb_npa = np.mean(
            embs_npa, axis=0, keepdims=True
        )  # embedding for '<unk>' token.
        # insert embeddings for pad and unk tokens at top of embs_npa.
        embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, embs_npa))
        word_to_idx = {word: i for i, word in enumerate(vocab_npa)}
        return embs_npa, word_to_idx

    def get_word_embedding(self, word):
        word = python_lowercase_remove_punctuation(word)
        if word in self.word_to_idx:
            return (
                self.embedding_layer(torch.tensor(self.word_to_idx[word]))
                .detach()
                .numpy()
            )
        else:
            # print(f"Word {word} not found in GloVe vocabulary. Returning <unk>.")
            return (
                self.embedding_layer(torch.tensor(self.word_to_idx["<unk>"]))
                .detach()
                .numpy()
            )
