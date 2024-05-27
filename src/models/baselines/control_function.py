from torch.utils.data import TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import Adam
import torch

from src.utils.torch_utils import MLPGaussianRegressor
from src.models.baselines.GloVe import GloVeModel
from src.models.baselines.fastText import FastTextModel

# Define the hyperparameters
H_PARAMS = {
    "num_layers": 3,
    "input_size": 300,  # Update this based on the word embedding model
    "hidden_size": 128,
    "num_labels": 1,
    "dropout_probability": 0.1,
    "learning_rate": 0.001,
    "batch_size": 32,
    "max_epochs": 10,
}


class MLPModule(pl.LightningModule):
    def __init__(self, hyperparams, dataset_train=None, dataset_val=None):
        super().__init__()
        self.hyperparams = hyperparams
        self.model = MLPGaussianRegressor(
            num_layers=self.hyperparams["num_layers"],
            input_size=self.hyperparams["input_size"],
            hidden_size=self.hyperparams["hidden_size"],
            num_labels=self.hyperparams["num_labels"],
            dropout_probability=self.hyperparams["dropout_probability"],
        )
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val

    def forward(self, batch, verbose=True):
        outputs = self.model(batch)
        if verbose:
            print(f"outputs shape = {outputs.shape}")
        # split last dimension into mu and sigma
        mu, sigma = torch.chunk(outputs, chunks=2, dim=-1)
        # ensure positivity of sigma
        sigma = torch.nn.functional.softplus(sigma)
        # add a small constant for numerical stability
        epsilon = 1e-7
        sigma = sigma + epsilon
        return mu.squeeze(-1), sigma.squeeze(-1)

    def step(self, batch):
        x, y = batch
        mu, sigma = self(x)
        dist = torch.distributions.Normal(mu, sigma)
        log_likelihood = dist.log_prob(y)
        negative_log_likelihood = -log_likelihood
        loss = negative_log_likelihood.mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.step(batch)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hyperparams["learning_rate"])

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.hyperparams["batch_size"])

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.hyperparams["batch_size"])

    def test_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma = self.model(x)
        dist = torch.distributions.Normal(mu, sigma)
        log_likelihood = dist.log_prob(y)
        negative_log_likelihood = -log_likelihood
        loss = negative_log_likelihood.mean()
        self.log("test_loss", loss)

    def test_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.hyperparams["batch_size"])


class ControlFunction:
    def __init__(
        self,
        word_embedding_type,
        word_embedding_path,
        hparams=H_PARAMS,
    ):
        self.hparams = hparams
        self.model = None

        # Initialize the correct word embedding model
        if word_embedding_type.lower() == "glove":
            self.word_embedding_model = GloVeModel(word_embedding_path)
        elif word_embedding_type.lower() == "fasttext":
            raise NotImplementedError
            self.word_embedding_model = FastTextModel(word_embedding_path)
        else:
            raise ValueError(
                "Invalid word embedding type. Must be either 'glove' or 'fasttext'."
            )

    def fit(self, words, labels, accelerator="mps"):
        # Convert the words to their corresponding embeddings
        print(f"Converting {len(words)} words to embeddings...")
        train_inputs = [
            torch.tensor(self.word_embedding_model.get_word_embedding(word))
            for word in words
        ]
        train_inputs = torch.stack(train_inputs)
        print(f"Done.")

        train_labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        # Create a dataset from the inputs and labels
        dataset_train = TensorDataset(train_inputs, train_labels)
        dataset_val = TensorDataset(
            train_inputs, train_labels
        )  # For simplicity, use train set as validation set.

        # Create an instance of the MLPModule and a trainer
        self.model = MLPModule(self.hparams, dataset_train, dataset_val)

        trainer = Trainer(
            max_epochs=self.hparams["max_epochs"], accelerator=accelerator
        )

        # Fit the model
        trainer.fit(self.model)

    def predict(self, words):
        # Convert the words to their corresponding embeddings
        test_inputs = [
            torch.tensor(self.word_embedding_model.get_word_embedding(word))
            for word in words
        ]
        test_inputs = torch.stack(test_inputs)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(test_inputs).numpy()

        return predictions
