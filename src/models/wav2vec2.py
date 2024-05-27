import inspect
import os

import torch
import torch.nn.functional as F
from lightning import LightningModule
from transformers import (
    Wav2Vec2ForPreTraining,
    Wav2Vec2Config,
    BertTokenizer,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)
from torchmetrics import MinMetric

from src.utils.torch_metrics import MeanMetric
from src.utils.utils import randomize_model
from src.utils.utils import get_device

DEVICE = get_device()


class Wav2Vec2Module(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer = AdamW,
        scheduler: torch.optim.lr_scheduler = get_linear_schedule_with_warmup,
        huggingface_model: str = "facebook/wav2vec2-base",
        use_pretrained: bool = True,
        mask_rate: float = 0.15,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.1,
        padding_val: float = 0.0,
        save_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        if use_pretrained:
            print(f"Using pretrained_model: {huggingface_model}")
            self.config = Wav2Vec2Config.from_pretrained(huggingface_model)
            self.model = Wav2Vec2ForPreTraining.from_pretrained(huggingface_model)
        else:
            print(f"Using randomly initialized model: {huggingface_model}")
            self.config = Wav2Vec2Config()
            self.model = Wav2Vec2ForPreTraining(self.config)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_loss_best = MinMetric()

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

    def compute_masks(self, input_values):
        batch_size, raw_sequence_length = input_values.shape
        sequence_length = self.model._get_feat_extract_output_lengths(
            raw_sequence_length
        ).item()

        mask_time_indices = _compute_mask_indices(
            shape=(batch_size, sequence_length),
            mask_prob=self.hparams.mask_rate,
            mask_length=2,
        )
        sampled_negative_indices = _sample_negative_indices(
            features_shape=(batch_size, sequence_length),
            num_negatives=self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        mask_time_indices = torch.tensor(
            data=mask_time_indices, device=input_values.device, dtype=torch.long
        )
        sampled_negative_indices = torch.tensor(
            data=sampled_negative_indices, device=input_values.device, dtype=torch.long
        )

        return mask_time_indices, sampled_negative_indices

    def forward(self, inputs, verbose=False):
        input_values = inputs["input_values"]
        mask_time_indices, sampled_negative_indices = self.compute_masks(input_values)
        outputs = self.model(
            input_values,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices,
            output_hidden_states=True,
        )
        return outputs

    def step(self, batch, batch_idx, verbose=False):
        inputs = batch

        if verbose:
            print(f"input_values: {inputs['input_values']}")
            print(f"attention_mask: {inputs['attention_mask']}")

        bs, seq_len = inputs["input_values"].shape
        outputs = self.forward(inputs)
        loss = outputs.loss / bs  # wave2vec2 loss is not averaged over batch
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)

        # update and log metrics
        self.val_loss(loss)
        self.log("test/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_epoch_end(self):
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
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
