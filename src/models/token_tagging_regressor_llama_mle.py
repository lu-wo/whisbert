import os
import inspect
from typing import Any, Dict, List, Tuple
import pickle

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MinMetric, MaxMetric
from torch.distributions import Normal
from torch.nn import L1Loss

from transformers import AdamW, AutoModel, get_linear_schedule_with_warmup
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer

from memory_profiler import profile

from src.utils import utils
from src.utils.torch_utils import (
    masked_loss,
    masked_GNLLL,
    MLPRegressor,
    freeze_pretrained_model,
    print_num_trainable_params,
)
from src.utils.torch_metrics import (
    MaskedMeanAbsoluteError,
    MaskedR2Score,
    MaskedPearsonCorrCoef,
    MeanMetric,
)


class TokenTaggingRegressorLlamaMLE(LightningModule):
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
        num_labels: int,
        freeze_lm: bool = False,
        train_last_k_layers: int = None,
        optimizer: torch.optim.Optimizer = AdamW,
        scheduler: torch.optim.lr_scheduler = get_linear_schedule_with_warmup,
        use_mlp: bool = False,
        p_dropout: float = 0.0,
        mlp_hidden_size: int = 512,
        mlp_num_layers: int = 5,
        loss_fn: nn.Module = None,
        output_activation: nn.Module = torch.nn.Identity(),
        save_path: str = None,
        llama_path: str = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["loss_fn", "output_activation"])

        # Load model and add head
        if llama_path is not None:
            print("Loading LLAMA model.")
            self.model = LlamaForCausalLM.from_pretrained(llama_path)
        else:
            print("Loading Huggingface model.")
            self.model = AutoModel.from_pretrained(huggingface_model)

        if freeze_lm:
            print("Freezing pretrained model.")
            for param in self.model.parameters():
                param.requires_grad = False

        if train_last_k_layers is not None:
            print(f"Freezing all but last {train_last_k_layers} layers.")
            self.model = freeze_pretrained_model(
                model=self.model, model_name=huggingface_model, k=train_last_k_layers
            )

        if use_mlp:
            print("Using MLP as head.")
            self.regressor = MLPRegressor(
                num_layers=mlp_num_layers,
                input_size=self.model.config.hidden_size,
                hidden_size=mlp_hidden_size,
                num_labels=num_labels * 2,
                dropout_probability=p_dropout,
            )
        else:
            print("Using linear layer as head.")
            self.regressor = nn.Linear(self.model.config.hidden_size, num_labels * 2)

        self.output_activation = output_activation

        # full==True: true negative log likelihood (with constant)
        self.loss_fn = nn.GaussianNLLLoss(full=True, reduction="none")

        # Init classifier weights according to initialization rules of model
        self.model._init_weights(self.regressor)

        # Apply dropout rate of model
        dropout_prob = p_dropout  # self.model.config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_loss = MeanMetric()  # already masked in loss function in step
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_mae = MaskedMeanAbsoluteError()
        self.val_mae = MaskedMeanAbsoluteError()
        self.test_mae = MaskedMeanAbsoluteError()

        # self.train_r2 = MaskedR2Score()
        self.val_r2 = MaskedR2Score()
        self.test_r2 = MaskedR2Score()

        self.val_pearson = MaskedPearsonCorrCoef()
        self.test_pearson = MaskedPearsonCorrCoef()

        # for logging best so far validation accuracy
        self.val_loss_best = MinMetric()
        self.val_mae_best = MinMetric()
        self.val_r2_best = MaxMetric()
        self.val_pearson_best = MaxMetric()

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

        # print number of trainable parameters
        print_num_trainable_params(
            self, model_name=f"TokenTaggingRegressorMLE {huggingface_model}"
        )

    def forward(self, batch: Dict[str, torch.tensor], eps=1e-4, verbose=False):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[
            -1
        ]  # access the last hidden state
        # outputs = self.dropout(outputs)
        outputs = self.regressor(outputs)
        if verbose:
            print(f"outputs shape {outputs.shape}")
        # split last dimension into mu and var
        mu, var = torch.chunk(outputs, chunks=2, dim=-1)
        if verbose:
            print(f"mu shape {mu.shape}, var shape {var.shape}")
        # ensure positivity of var
        var = torch.nn.functional.softplus(var)
        # add a small constant for numerical stability
        var = (var + eps).squeeze(-1)

        # have to squeeze the last dimension due to chunking
        if self.output_activation is not None:
            mu = self.output_activation(mu.squeeze(-1))
        return mu, var

    def on_train_start(self):
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.val_r2.reset()
        self.val_r2_best.reset()

    # @profile
    def step(self, batch: Dict[str, torch.tensor], verbose: bool = False):
        mu, var = self(batch)  # forward pass
        labels = batch["tokenized_labels"]
        loss_mask = batch["loss_mask"]  # ignore padded sequence in loss

        if verbose:
            print(f"text: {batch['input_text']}")
            print(f"mu {mu}, \nvar {var}, \nlabels {labels}, \nmask {loss_mask}")
        # print(f"text: {batch['input_text']}")
        # print(f"tokenized labels: {labels}")
        # print(f"word to tokens: {batch['word_to_tokens']}")
        masked_gaussian_nll = masked_GNLLL(
            input=mu, target=labels, var=var, mask=loss_mask, loss_fn=self.loss_fn
        )
        masked_mae = masked_loss(
            labels=labels,
            predictions=mu,
            mask=loss_mask,
            loss_fn=L1Loss(reduction="none"),
        )
        if verbose:
            print(f"masked neg log_likelihood loss: {masked_gaussian_nll}")
            print(f"masked mae: {masked_mae}")

        return masked_gaussian_nll, mu, var

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        # loss is batch loss, train_loss tracks it over the epoch
        loss, preds, var = self.step(batch)
        self.train_loss(loss)
        self.train_mae(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/mae", self.train_mae, on_step=True, on_epoch=True, prog_bar=True
        )
        return {
            "loss": loss,
            "predictions": preds,
            "targets": batch["tokenized_labels"],
            "attention_mask": batch["attention_mask"],
        }

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        # loss is batch loss, val_loss tracks it over the epoch
        loss, preds, var = self.step(batch)
        # self.val_loss(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.val_loss(loss)
        self.val_mae(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.val_r2(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.val_pearson(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/pearson",
            self.val_pearson,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {
            "loss": loss,
            "preds": preds,
            "targets": batch["tokenized_labels"],
            "attention_mask": batch["attention_mask"],
        }

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)
        mae = self.val_mae.compute()
        self.val_mae_best(mae)
        self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True)
        r2 = self.val_r2.compute()
        self.val_r2_best(r2)
        self.log("val/r2_best", self.val_r2_best.compute(), prog_bar=True)
        pearson = self.val_pearson.compute()
        self.val_pearson_best(pearson)
        self.log("val/pearson_best", self.val_pearson_best.compute(), prog_bar=True)

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        # loss is batch loss, test_loss tracks it over the epoch
        loss, preds, var = self.step(batch)
        # self.test_loss(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.test_mae(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)
        # self.test_r2(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.test_r2(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.log("test/r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=True)
        # self.test_pearson(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.test_pearson(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.log(
            "test/pearson",
            self.test_pearson,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # TODO: make callback work
        # save predictions
        np.save(
            f"{self.save_path}/predictions/test_input_ids_{batch_idx}.npy",
            batch["input_ids"].cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/test_attention_mask_{batch_idx}.npy",
            batch["attention_mask"].cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/test_labels_{batch_idx}.npy",
            batch["tokenized_labels"].cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/test_preds_mu{batch_idx}.npy",
            preds.cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/test_preds_var{batch_idx}.npy",
            var.cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/test_loss_mask_{batch_idx}.npy",
            batch["loss_mask"].cpu().numpy(),
        )
        # pickle input text and original labels and word_to_tokens for later evaluation
        with open(
            f"{self.save_path}/predictions/test_input_text_{batch_idx}.pkl", "wb"
        ) as f:
            pickle.dump(batch["input_text"], f)
        with open(
            f"{self.save_path}/predictions/test_original_labels_{batch_idx}.pkl", "wb"
        ) as f:
            pickle.dump(batch["original_labels"], f)
        with open(
            f"{self.save_path}/predictions/test_word_to_tokens_{batch_idx}.pkl", "wb"
        ) as f:
            pickle.dump(batch["word_to_tokens"], f)
        #
        #

        return {
            "loss": loss,
            "preds": preds,
            "targets": batch["tokenized_labels"],
            "attention_mask": batch["attention_mask"],
            "loss_mask": batch["loss_mask"],
            "input_ids": batch["input_ids"],
            "input_text": batch["input_text"],
            "original_labels": batch["original_labels"],
        }

    def on_test_epoch_end(self):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()

        self.train_mae.reset()
        self.test_mae.reset()
        self.val_mae.reset()

        self.test_r2.reset()
        self.val_r2.reset()

        self.val_pearson.reset()
        self.test_pearson.reset()

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
