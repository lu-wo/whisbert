import inspect
import os
import random

import torch
import torch.nn.functional as F
from lightning import LightningModule
from transformers import (
    BertConfig,
    BertForPreTraining,
    BertTokenizer,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torchmetrics import MinMetric, MaxMetric

from src.utils.torch_metrics import MeanMetric
from src.utils.utils import randomize_model
from src.utils.utils import get_device
from src.models.components.prosody_encoder import GeneralEncoder

DEVICE = get_device()


class BertMLMTrainingModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer = AdamW,
        scheduler: torch.optim.lr_scheduler = get_linear_schedule_with_warmup,
        huggingface_model: str = "bert-base-cased",
        use_pretrained: bool = False,
        #
        prosody_encoder_path: str = None,
        prosody_feature_dim: int = 20,
        #
        text_encoder_path: str = None,
        tokenizer_path: str = None,
        mask_rate: float = 0.15,
        vocab_size: int = 30522,
        type_vocab_size: int = 2,
        max_position_embeddings: int = 512,
        position_embedding_type: str = "absolute",
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.1,
        pad_token_id: int = 0,
        save_path: str = None,
        divide_l2_by: float = 10000.0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        if use_pretrained:
            print(f"Using FLAVA pretrained text model: {huggingface_model}")
            self.text_model = BertForPreTraining.from_pretrained(
                "bert-base-cased",
            )
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

            # TODO: load pretrained prosody encoder and multimodal encoder
            pass
        else:
            print("Using randomly initialized FLAVA text model")
            # TODO: optionally init other model configs via FlavaTextConfig
            self.text_config = BertConfig(
                vocab_size=vocab_size,
                type_vocab_size=type_vocab_size,
                max_position_embeddings=max_position_embeddings,
                position_embedding_type=position_embedding_type,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                pad_token_id=pad_token_id,
            )
            self.text_model = BertForPreTraining(self.config)
            if tokenizer_path is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

            self.prosody_encoder = GeneralEncoder(
                feature_dim=prosody_feature_dim,
                intermediate_dim=hidden_size,
                nhead=num_attention_heads,
                num_layers=num_hidden_layers,
            )

            self.multimodal_encoder = GeneralEncoder(
                feature_dim=hidden_size,
                intermediate_dim=hidden_size,
                nhead=num_attention_heads,
                num_layers=num_hidden_layers,
            )

        # multimodal heads
        self.text_mm_projection = torch.nn.Linear(hidden_size, hidden_size)
        self.prosody_mm_projection = torch.nn.Linear(hidden_size, hidden_size)

        # MLM heads
        self.text_mlm_head = torch.nn.Linear(hidden_size, vocab_size)

        print(f"Model tokenizer pad token id: {self.tokenizer.pad_token_id}")
        print(f"Model tokenizer mask token id: {self.tokenizer.mask_token_id}")

        # for averaging loss across batches
        self.train_mlm_loss = MeanMetric()
        self.val_mlm_loss = MeanMetric()
        self.test_mlm_loss = MeanMetric()
        self.val_mlm_loss_best = MinMetric()

        # multimodal losses
        self.train_gc_loss = MeanMetric()
        self.val_gc_loss = MeanMetric()
        self.test_gc_loss = MeanMetric()
        self.val_gc_loss_best = MinMetric()

        self.train_mmm_loss = MeanMetric()
        self.val_mmm_loss = MeanMetric()
        self.test_mmm_loss = MeanMetric()
        self.val_mmm_loss_best = MinMetric()

        # for averaging PLL across batches
        # self.val_pll = MeanMetric()
        self.test_pll = MeanMetric()
        # self.val_pll_best = MaxMetric()

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

    def _mask_text_tokens(self, inputs, attention_mask=None, mask_prob=0.3):
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

    def text_forward(self, input_ids_masked, attention_mask, token_type_ids, labels):
        # we only do MLM, therefore next_sentence_label is always 0
        batch_size = input_ids_masked.shape[0]
        next_sentence_label = torch.zeros(batch_size, dtype=torch.long).to(
            input_ids_masked.device
        )
        return self.model(
            input_ids=input_ids_masked,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            next_sentence_label=next_sentence_label,
        )

    def _unimodal_step(self, text_inputs, batch_idx, verbose=False):
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]
        token_type_ids = text_inputs["token_type_ids"]
        input_ids_masked, labels = self._mask_text_tokens(
            input_ids,
            attention_mask=attention_mask,
            mask_prob=self.hparams.mask_rate,
        )
        outputs = self.text_forward(
            input_ids_masked, attention_mask, token_type_ids, labels
        )
        return outputs.loss  # MLM loss

    def _global_contrastive_loss(
        self, prosody_clf_tokens, text_clf_tokens, temperature=0.1
    ):
        # Linearly project into an embedding space
        prosody_proj = self.prosody_mm_projection(prosody_clf_tokens)
        text_proj = self.text_mm_projection(text_clf_tokens)

        # L2-normalization
        prosody_proj = F.normalize(prosody_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)

        # Dot-product to compute similarities
        logits = torch.matmul(prosody_proj, text_proj.t())

        # Create labels for the matched pairs
        labels = torch.arange(logits.shape[0]).to(logits.device)

        # Apply softmax loss scaled by temperature
        loss = F.cross_entropy(logits / temperature, labels)

        return loss

    def _masked_mmm_loss(
        self, prosody_masked_mm_encodings, text_masked_mm_encodings, text_labels, mask
    ):
        prosody_mask = mask[:, : len(prosody_masked_mm_encodings[0]), :].float()
        text_mask = mask[:, len(prosody_masked_mm_encodings[0]) :, :].float()

        # L2 loss for prosody embeddings
        prosody_reconstruction_loss = F.mse_loss(
            prosody_masked_mm_encodings,
            self.prosody_mm_projection(prosody_masked_mm_encodings),
        )
        prosody_reconstruction_loss = (
            prosody_reconstruction_loss * prosody_mask
        ).mean()

        # Cross-entropy loss for text embeddings
        # Assuming text_masked_mm_encodings are the true labels
        text_logits = self.text_mlm_head(text_masked_mm_encodings)
        text_mlm_loss = F.cross_entropy(
            text_logits.transpose(1, 2), text_labels, ignore_index=-100
        )
        text_mlm_loss = (text_mlm_loss * text_mask).mean()

        return prosody_reconstruction_loss / self.hparams.divide_l2_by + text_mlm_loss

    def _get_multimodal_masked_inputs(
        self, prosody_mm_encodings, text_mm_encodings, mask_prob=0.3, mask_value=-100
    ):
        # Get a binary mask of elements to mask
        mask = torch.full_like(
            prosody_mm_encodings, fill_value=mask_prob
        ) > torch.rand_like(prosody_mm_encodings)

        # Clone the input to avoid modifying it directly
        prosody_masked = prosody_mm_encodings.clone().detach()
        text_masked = text_mm_encodings.clone().detach()

        # Apply the mask
        prosody_masked[mask] = mask_value
        text_masked[mask] = mask_value

        # Concatenate prosody and text encodings
        masked_encodings = torch.cat((prosody_masked, text_masked), dim=0)

        return masked_encodings, mask

    def _masked_multimodal_pass(
        self, prosody_encodings, text_encodings, batch_idx, verbose=False
    ):
        masked_seq, mask = self._get_multimodal_masked_inputs(
            prosody_encodings, text_encodings
        )
        mm_encodings = self.multimodal_encoder(masked_seq)
        return mm_encodings, mask

    def _multimodal_pass(
        self, prosody_encodings, text_encodings, batch_idx, verbose=False
    ):
        return self.multimodal_encoder(
            torch.cat([prosody_encodings, text_encodings], dim=0)
        )

    def _multimodal_step(self, prosody_inputs, text_inputs, batch_idx, verbose=False):
        prosody_encodings = self.prosody_encoder(prosody_inputs)
        prosody_clf_token = prosody_encodings[:, 0, :]
        text_encodings = self.text_model(**text_inputs).hidden_states[-1]
        text_clf_token = text_encodings[:, 0, :]

        # compute multimodal encodings:
        multimodal_encodings = self._multimodal_pass(
            prosody_encodings, text_encodings, batch_idx, verbose=verbose
        )
        multimodal_clf_token = multimodal_encodings[:, 0, :]
        prosody_mm_encodings = multimodal_encodings[:, 1 : len(prosody_encodings), :]
        prosody_mm_clf_token = prosody_mm_encodings[:, 0, :]
        text_mm_encodings = multimodal_encodings[len(prosody_encodings) :, :, :]
        text_mm_clf_token = text_mm_encodings[:, 0, :]

        # compute masked multimodal encodings
        masked_multimodal_encodings, mask = self._masked_multimodal_pass(
            prosody_encodings, text_encodings, batch_idx, verbose=verbose
        )
        masked_multimodal_clf_token = masked_multimodal_encodings[:, 0, :]
        prosody_masked_mm_encodings = masked_multimodal_encodings[
            :, 1 : len(prosody_encodings), :
        ]
        text_masked_mm_encodings = masked_multimodal_encodings[
            len(prosody_encodings) :, :, :
        ]

        # compute global contrastive loss
        gc_loss = self._global_contrastive_loss(
            prosody_mm_clf_token, text_mm_clf_token, batch_idx, verbose=verbose
        )

        # compute masked multimodal modeling loss
        mmm_loss = self._mmm_loss(
            prosody_masked_mm_encodings, text_masked_mm_encodings, mask, batch_idx
        )

        return gc_loss, mmm_loss

    def compute_pll(self, text_inputs):
        """
        Compute the average Pseudo Log-Likelihood (PLL) for a batch of token sequences.

        Args:
            token_sequences (torch.Tensor): Tensor of token sequences of shape (batch_size, seq_length).
            attention_mask (torch.Tensor): Tensor indicating valid entries of shape (batch_size, seq_length).

        Returns:
            avg_pll (torch.Tensor): The average Pseudo Log-Likelihood of the token sequences under the model.
        """
        token_sequences = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        batch_size, seq_length = token_sequences.shape
        log_probs = []

        for i in range(seq_length):
            # Create a copy of the token sequences with the current token masked
            masked_tokens = token_sequences.clone()
            masked_tokens[:, i] = self.tokenizer.mask_token_id

            # Compute the model's log probabilities for the token sequences
            outputs = self.model(
                input_ids=token_sequences,
                input_ids_masked=masked_tokens,
                attention_mask=attention_mask,
            )
            predictions = outputs.mlm_logits
            log_prob = torch.nn.functional.log_softmax(predictions, dim=-1)

            # Select the log probability of the original token at the masked position
            original_tokens = token_sequences[:, i]
            token_log_prob = log_prob[range(batch_size), i, original_tokens]

            # Mask out invalid entries
            valid = attention_mask[:, i]
            # multiply by valid to zero out invalid
            token_log_prob = token_log_prob * valid
            log_probs.append(token_log_prob)

        # Sum the log probabilities to compute the PLL
        pll = torch.stack(log_probs).sum()

        # Compute the average PLL by dividing the PLL by the batch size
        avg_pll = pll / batch_size

        return avg_pll

    def training_step(self, batch, batch_idx):
        prosody_inputs, text_inputs = batch

        # 50% multimodal, 50% unimodal
        if random.random() < 0.5:
            loss_gc, loss_mmm = self._multimodal_step(
                prosody_inputs, text_inputs, batch_idx
            )
            self.log(
                "train/loss_gc", loss_gc, on_step=True, on_epoch=True, prog_bar=True
            )
            self.log(
                "train/loss_mmm", loss_mmm, on_step=True, on_epoch=True, prog_bar=True
            )
            loss = loss_gc + loss_mmm
            self.train_mmm_loss(loss_mmm)
            self.train_gc_loss(loss_gc)
        else:
            loss = self._unimodal_step(text_inputs, batch_idx)
            self.train_mlm_loss(loss)
            self.log(
                "train/loss_mlm",
                self.train_mlm_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        prosody_inputs, text_inputs = batch

        # 50% multimodal, 50% unimodal
        if random.random() < 0.5:
            loss_gc, loss_mmm = self._multimodal_step(
                prosody_inputs, text_inputs, batch_idx
            )
            self.log("val/loss_gc", loss_gc, on_step=True, on_epoch=True, prog_bar=True)
            self.log(
                "val/loss_mmm", loss_mmm, on_step=True, on_epoch=True, prog_bar=True
            )
            loss = loss_gc + loss_mmm
            self.val_mmm_loss(loss_mmm)
            self.val_gc_loss(loss_gc)
        else:
            loss = self._unimodal_step(text_inputs, batch_idx)
            self.val_mlm_loss(loss)
            self.log(
                "val/loss_mlm",
                self.train_mlm_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def on_validation_epoch_end(self):
        val_mlm_loss = self.val_mlm_loss.compute()
        val_mmm_loss = self.val_mmm_loss.compute()
        val_gc_loss = self.val_gc_loss.compute()
        self.log("val/loss_mlm", val_mlm_loss)
        self.log("val/loss_mmm", val_mmm_loss)
        self.log("val/loss_gc", val_gc_loss)
        self.val_mlm_loss.reset()
        self.val_mmm_loss.reset()
        self.val_gc_loss.reset()

    def test_step(self, batch, batch_idx):
        prosody_inputs, text_inputs = batch

        # 50% multimodal, 50% unimodal
        if random.random() < 0.5:
            loss_gc, loss_mmm = self._multimodal_step(
                prosody_inputs, text_inputs, batch_idx
            )
            self.log(
                "test/loss_gc", loss_gc, on_step=True, on_epoch=True, prog_bar=True
            )
            self.log(
                "test/loss_mmm", loss_mmm, on_step=True, on_epoch=True, prog_bar=True
            )
            loss = loss_gc + loss_mmm
            self.test_mmm_loss(loss_mmm)
            self.test_gc_loss(loss_gc)
        else:
            loss = self._unimodal_step(text_inputs, batch_idx)
            self.test_mlm_loss(loss)
            self.log(
                "test/loss_mlm",
                self.train_mlm_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        # additionally evaluate pseudo log likelihood
        if batch_idx % 500 == 0:
            pll = self._pseudo_log_likelihood(text_inputs)
            self.log("test/pll", pll, on_step=True, on_epoch=True, prog_bar=True)
            self.test_pll(pll)

        return loss

    def on_epoch_end(self):
        test_mlm_loss = self.test_mlm_loss.compute()
        test_mmm_loss = self.test_mmm_loss.compute()
        test_gc_loss = self.test_gc_loss.compute()
        self.log("test/loss_mlm", test_mlm_loss)
        self.log("test/loss_mmm", test_mmm_loss)
        self.log("test/loss_gc", test_gc_loss)
        self.test_mlm_loss.reset()
        self.test_mmm_loss.reset()
        self.test_gc_loss.reset()

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
