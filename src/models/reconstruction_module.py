import os
import inspect
from typing import Any, Dict, List, Tuple
import pickle

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MinMetric, MaxMetric
from transformers import AdamW, AutoModel, get_linear_schedule_with_warmup
import numpy as np

from src.utils import utils
from src.models.components.seq2_autoencoder import Seq2SeqAutoencoder
from src.utils.torch_utils import masked_loss
from src.utils.torch_metrics import (
    MaskedMeanSquaredError,
    MaskedR2Score,
    MaskedPearsonCorrCoef,
    MaskedSpearmanCorrCoeff,
)


class ReconstructionModule(LightningModule):
    """
    Transformer Model for Token Tagging, i.e. per Token Sequence Regression.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        huggingface_model: str,
        input_features: int = 1,
        hidden_size: int = 128,
        output_features: int = 1,
        num_layers: int = 1,
        bidirectional: bool = False,
        teacher_forcing_ratio: float = 0.5,
        optimizer: torch.optim.Optimizer = AdamW,
        scheduler: torch.optim.lr_scheduler = get_linear_schedule_with_warmup,
        p_dropout: float = 0.1,
        loss_fn: nn.Module = torch.nn.MSELoss(reduction="none"),
        output_activation: nn.Module = torch.nn.Identity(),
        save_path: str = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["loss_fn", "output_activation"])

        # Load model and add head
        self.model = Seq2SeqAutoencoder(
            input_size=input_features,
            hidden_size=hidden_size,
            output_size=output_features,
            num_layers=num_layers,
            dropout=p_dropout,
            bidirectional=bidirectional,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        self.output_activation = output_activation

        # Apply dropout rate of model
        dropout_prob = p_dropout  # self.model.config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        # loss function (assuming single-label multi-class classification)
        self.loss_fn = loss_fn

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_loss = MaskedMeanSquaredError()
        self.val_loss = MaskedMeanSquaredError()
        self.test_loss = MaskedMeanSquaredError()

        # self.train_r2 = MaskedR2Score()
        self.val_r2 = MaskedR2Score()
        self.test_r2 = MaskedR2Score()

        # self.train_pearson = MaskedPearsonCorrCoef()
        self.val_pearson = MaskedPearsonCorrCoef()
        self.test_pearson = MaskedPearsonCorrCoef()

        # self.train_spearman = MaskedSpearmanCorrCoeff()
        # self.val_spearman = MaskedSpearmanCorrCoeff()
        # self.test_spearman = MaskedSpearmanCorrCoeff()

        # for logging best so far validation accuracy
        self.val_loss_best = MinMetric()
        self.val_r2_best = MaxMetric()
        self.val_pearson_best = MaxMetric()
        # self.val_spearman_best = MaxMetric()

        # Collect the forward signature
        params = inspect.signature(self.model.forward).parameters.values()
        params = [
            param.name for param in params if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        self.forward_signature = params

        # create save path dir
        if save_path is not None:
            self.save_path = save_path
            os.makedirs(os.path.join(self.save_path, "predictions"), exist_ok=True)

    def forward(self, batch: Dict[str, torch.tensor]):
        return self.model(batch)

    def step(self, batch: Dict[str, torch.tensor]):
        padded_sequences, lengths, mask = batch
        x_rec, latent = self(padded_sequences)  # forward pass
        loss = masked_loss(
            labels=padded_sequences, predictions=x_rec, mask=mask, loss_fn=self.loss_fn
        )
        return loss, latent, padded_sequences, x_rec, mask

    def on_train_start(self):
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.val_r2.reset()
        self.val_r2_best.reset()

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        # loss is batch loss, train_loss tracks it over the epoch
        loss, latent, x, x_rec, mask = self.step(batch)

        self.train_loss(x_rec, x, mask)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        # self.train_r2(x_rec, x, mask)
        # self.log("train/r2", self.train_r2, on_step=True, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
            "latent": latent,
            "preds": x_rec,
            "targets": x,
            "mask": mask,
        }

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        # loss is batch loss, val_loss tracks it over the epoch
        loss, latent, x, x_rec, mask = self.step(batch)

        self.val_loss(x_rec, x, mask)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_r2(x_rec, x, mask)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)

        self.val_pearson(x_rec, x, mask)
        self.log(
            "val/pearson",
            self.val_pearson,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # self.val_spearman(x_rec, x, mask)
        # self.log(
        #     "val/spearman",
        #     self.val_spearman,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        return {
            "loss": loss,
            "latent": latent,
            "preds": x_rec,
            "targets": x,
            "mask": mask,
        }

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)
        r2 = self.val_r2.compute()
        self.val_r2_best(r2)
        self.log("val/r2_best", self.val_r2_best.compute(), prog_bar=True)
        pearson = self.val_pearson.compute()
        self.val_pearson_best(pearson)
        self.log("val/pearson_best", self.val_pearson_best.compute(), prog_bar=True)
        # spearman = self.val_spearman.compute()
        # self.val_spearman_best(spearman)
        # self.log("val/spearman_best", self.val_spearman_best.compute(), prog_bar=True)

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        # loss is batch loss, test_loss tracks it over the epoch
        loss, latent, x, x_rec, mask = self.step(batch)
        # print(f"x: {x}")
        # print(f"x_rec: {x_rec}")
        # print(f"loss: {loss}")
        self.test_loss(x_rec, x, mask)
        self.test_r2(x_rec, x, mask)
        self.test_pearson(x_rec, x, mask)
        # self.test_spearman(x_rec, x, mask)
        # print(f"test_spearman {self.test_spearman.compute()}")
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/pearson",
            self.test_pearson,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # self.log(
        #     "test/spearman",
        #     self.test_spearman,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

        # TODO: make callback work
        # save predictions
        np.save(
            f"{self.save_path}/predictions/targets_{batch_idx}.npy",
            x.cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/latents_{batch_idx}.npy",
            latent.cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/predictions_{batch_idx}.npy",
            x_rec.cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/loss_masks_{batch_idx}.npy",
            mask.cpu().numpy(),
        )

        return {
            "loss": loss,
            "preds": x_rec,
            "targets": x,
            "loss_mask": mask,
        }

    def on_test_epoch_end(self):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        self.train_loss.reset()
        self.test_loss.reset()
        self.val_loss.reset()
        self.train_r2.reset()
        self.test_r2.reset()
        self.val_r2.reset()
        self.val_pearson.reset()
        self.test_pearson.reset()
        # self.val_spearman.reset()
        # self.test_spearman.reset()A

    @property
    def total_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if (
            isinstance(self.trainer.limit_train_batches, int)
            and self.trainer.limit_train_batches != 0
        ):
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (
            dataset_size // effective_batch_size
        ) * self.trainer.max_epochs

        if self.trainer.max_steps and 0 < self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
