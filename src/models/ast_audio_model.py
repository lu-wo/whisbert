import inspect
import os

import torch
import torch.nn.functional as F
from lightning import LightningModule
from transformers import (
    FlavaForPreTraining,
    BertTokenizer,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torchmetrics import MinMetric

from src.utils.torch_metrics import MeanMetric
from src.utils.utils import randomize_model
from src.utils.utils import get_device

DEVICE = get_device()


class ASTAudioModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer = AdamW,
        scheduler: torch.optim.lr_scheduler = get_linear_schedule_with_warmup,
        huggingface_model: str = "facebook/flava-full",
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
            print(f"Using FLAVA pretrained text model: {huggingface_model}")
            self.model = FlavaForPreTraining.from_pretrained(
                "facebook/flava-full",
            )
        else:
            print("Using randomly initialized FLAVA text model")
            # TODO: optionally init other model configs via FlavaTextConfig
            self.model = FlavaForPreTraining.from_pretrained(
                "facebook/flava-full",
            )
            self.model = randomize_model(self.model)

        print(f"Model tokenizer pad token id: {self.tokenizer.pad_token_id}")
        print(f"Model tokenizer mask token id: {self.tokenizer.mask_token_id}")

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

    def _mask_tokens(self, inputs, attention_mask=None, mask_prob=0.3):
        labels = inputs.clone()
        inputs = inputs.clone().to(DEVICE)

        # Probability matrix should only allow masking where attention_mask is 1
        if attention_mask is not None:
            # Use the attention_mask to limit where tokens can be masked
            probability_matrix = (
                torch.full(labels.shape, mask_prob).to(DEVICE) * attention_mask
            )
        else:
            # If no attention_mask is provided, tokens can be masked anywhere
            probability_matrix = torch.full(labels.shape, mask_prob).to(DEVICE)

        # Determine which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Mask tokens
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids("[MASK]")

        # Replace -100 in labels that we do not want to compute the loss for
        labels[~masked_indices] = -100

        return inputs, labels

    def forward(self, inputs, verbose=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        input_ids_masked, labels = self._mask_tokens(
            input_ids,
            attention_mask=attention_mask,
            mask_prob=self.hparams.mask_rate,
        )

        if verbose:
            print(f"Input ids: {input_ids}")
            print(f"Attention mask: {attention_mask}")
            print(f"Token type ids: {token_type_ids}")
            print(f"Input ids masked: {input_ids_masked}")
            print(f"Labels: {labels}")

        return self.model(
            input_ids_masked=input_ids_masked,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            mlm_labels=labels,
        )

    def step(self, batch, batch_idx, verbose=False):
        if verbose:
            print(f"Batch: {batch}")
        outputs = self.forward(batch)
        if verbose:
            print(f"Outputs: {outputs}")
        loss = outputs.loss
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
