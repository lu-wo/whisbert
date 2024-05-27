from torch.utils.data import Dataset, DataLoader
import pickle
import os
import torch


class EmbeddingDataset(Dataset):
    def __init__(self, dir_path, mode, word_model):
        self.features = []
        self.labels = []
        self.mode = mode
        self.word_model = word_model

        with open(os.path.join(dir_path, f"{mode}_words.pkl"), "rb") as f:
            self.features = pickle.load(f)
        with open(os.path.join(dir_path, f"{mode}_labels.pkl"), "rb") as f:
            self.labels = pickle.load(f)

        # convert to float 32 tensors
        self.labels = torch.tensor(self.labels).float()

        print(
            f"shapes of features and labels: {len(self.features)}, {self.labels.shape}"
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # print(f"get item: {self.features[idx]}, {self.labels[idx]}")
        return self.word_model.get_word_embedding(self.features[idx]), self.labels[idx]
