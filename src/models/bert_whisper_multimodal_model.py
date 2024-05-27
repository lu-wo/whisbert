import inspect
import os, sys
import random
import time
from copy import deepcopy
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from lightning import LightningModule
from transformers import (
    BertConfig,
    BertForPreTraining,
    BertForMaskedLM,
    BertTokenizer,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torchmetrics import MinMetric, MaxMetric
from transformers import WhisperConfig, WhisperModel


from src.utils.torch_metrics import MeanMetric
from src.utils.utils import get_device
from src.utils.hf_utils import MultimodalModelOutput
from src.models.components.encoders import MyWhisperEncoder, GeneralEncoder
from src.utils.torch_utils import Wav2Vec2ContrastiveLoss

DEVICE = get_device()


class BertWhisperTrainingModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer = AdamW,
        scheduler: torch.optim.lr_scheduler = get_linear_schedule_with_warmup,
        huggingface_model: str = "bert-base-cased",
        use_pretrained: bool = False,
        tokenizer_path: str = None,
        uni_text_mask_rate: float = 0.15,
        uni_audio_mask_rate: float = 0.5,
        mm_text_mask_rate: float = 0.15,
        mm_audio_mask_rate: float = 0.5,
        vocab_size: int = 30522,
        type_vocab_size: int = 2,
        max_position_embeddings: int = 512,
        position_embedding_type: str = "absolute",
        hidden_size: int = 1536,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.1,
        # pad_token_id: int = 0,
        save_path: str = None,
        # whisper params
        audio_encoder_layers: int = 6,
        audio_encoder_heads: int = 6,
        audio_encoder_ff_dim: int = 1024,
        audio_encoder_max_source_positions: int = 4096,
        audio_encoder_num_mel_bins: int = 80,
        audio_encoder_kernel_size: int = 5,
        audio_encoder_stride: int = 3,
        audio_encoder_path: str = None,
        # loss param
        unimodal_prob: float = 0.5,
        mlm_weight: float = 0.0,
        mam_weight: float = 0.0,
        global_contrastive_weight: float = 0.0,
        mmm_text_weight: float = 0.0,
        mmm_audio_weight: float = 0.0,
        audio_text_matching_weight: float = 0.0,
        audio_loss: str = "l2",  # contrastive, l2
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        if tokenizer_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

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
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.text_encoder = BertForMaskedLM(self.text_config)

        # audio encoder
        if audio_encoder_path is not None:
            print(f"Loading audio encoder from {audio_encoder_path}")
            self.audio_config = WhisperConfig.from_pretrained(
                os.path.join(audio_encoder_path, "config.json")
            )
            self.audio_encoder = MyWhisperEncoder(self.audio_config)
            self.audio_encoder.load_state_dict(
                torch.load(os.path.join(audio_encoder_path, "pytorch_model.bin"))
            )
        else:
            self.audio_config = WhisperConfig(
                encoder_layers=audio_encoder_layers,
                encoder_attention_heads=audio_encoder_heads,
                encoder_ff_dim=audio_encoder_ff_dim,
                max_source_positions=audio_encoder_max_source_positions,
                num_mel_bins=audio_encoder_num_mel_bins,
                d_model=hidden_size,
                kernel_size=audio_encoder_kernel_size,
                stride=audio_encoder_stride,
                mask_rate=uni_audio_mask_rate,
                mask_value=-5,
                loss_name=audio_loss,
            )
            self.audio_encoder = MyWhisperEncoder(self.audio_config)

        self.multimodal_encoder = GeneralEncoder(
            feature_dim=hidden_size,
            intermediate_dim=hidden_size,
            nhead=num_attention_heads,
            num_layers=num_hidden_layers,
        )

        # uni to multi heads
        self.audio_to_mm_projection = torch.nn.Linear(hidden_size, hidden_size)
        self.text_to_mm_projection = torch.nn.Linear(hidden_size, hidden_size)

        # multimodal prediction heads
        self.text_mm_mlm_head = torch.nn.Linear(
            hidden_size, self.text_encoder.config.vocab_size
        )
        self.audio_mm_mam_head = torch.nn.Linear(hidden_size, hidden_size)

        self.text_clf_projection = torch.nn.Linear(hidden_size, hidden_size)
        self.audio_clf_projection = torch.nn.Linear(hidden_size, hidden_size)

        # multimodal matching head
        self.audio_text_matching_head = torch.nn.Linear(hidden_size, 2)  # two classes

        # unimodal heads
        # BERTForMaskedLM has already one
        # For Whisper we just use the conv embeddings

        print(f"Model tokenizer pad token id: {self.tokenizer.pad_token_id}")
        print(f"Model tokenizer mask token id: {self.tokenizer.mask_token_id}")

        # unimodal metrics
        self.train_uni_mlm_loss = MeanMetric()
        self.val_uni_mlm_loss = MeanMetric()
        self.test_uni_mlm_loss = MeanMetric()
        self.val_uni_mlm_loss_best = MinMetric()

        self.train_uni_mam_loss = MeanMetric()
        self.val_uni_mam_loss = MeanMetric()
        self.test_uni_mam_loss = MeanMetric()
        self.val_uni_mam_loss_best = MinMetric()

        # early stopping loss is for now unimodal mlm loss
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()

        # multimodal losses
        self.train_gc_loss = MeanMetric()
        self.val_gc_loss = MeanMetric()
        self.test_gc_loss = MeanMetric()
        self.val_gc_loss_best = MinMetric()

        self.train_mm_mam_loss = MeanMetric()
        self.val_mm_mam_loss = MeanMetric()
        self.test_mm_mam_loss = MeanMetric()
        self.val_mm_mam_loss_best = MinMetric()

        self.train_mm_mlm_loss = MeanMetric()
        self.val_mm_mlm_loss = MeanMetric()
        self.test_mm_mlm_loss = MeanMetric()
        self.val_mm_mlm_loss_best = MinMetric()

        self.train_mlm_text_loss = MeanMetric()
        self.val_mlm_text_loss = MeanMetric()
        self.test_mlm_text_loss = MeanMetric()
        self.val_mlm_text_loss_best = MinMetric()

        self.train_audio_text_matching_loss = MeanMetric()
        self.val_audio_text_matching_loss = MeanMetric()
        self.test_audio_text_matching_loss = MeanMetric()
        self.val_audio_text_matching_loss_best = MinMetric()

        # for averaging PLL across batches
        # self.val_pll = MeanMetric()
        self.test_pll = MeanMetric()
        # self.val_pll_best = MaxMetric()

        # create save path dir
        if save_path is not None:
            self.save_path = save_path
            os.makedirs(os.path.join(self.save_path, "predictions"), exist_ok=True)

    def forward(
        self,
        audio_inputs: Optional[torch.Tensor] = None,
        text_inputs: Optional[torch.Tensor] = None,
        text_labels: Optional[torch.Tensor] = None,
        skip_multimodal_encoder: Optional[bool] = None,
        output_audio_loss: Optional[bool] = True,
    ):
        audio_embeddings = None
        audio_mm_projection = None
        audio_conv_embeddings = None
        audio_loss = None
        audio_mask = None
        if audio_inputs is not None:
            audio_outputs = self.audio_encoder(
                **audio_inputs, output_loss=output_audio_loss
            )
            audio_embeddings = audio_outputs.last_hidden_state
            audio_conv_embeddings = audio_outputs.conv_embeddings
            audio_mask = audio_outputs.mask
            audio_mm_projection = self.audio_to_mm_projection(audio_embeddings)
            audio_loss = audio_outputs.loss
            print(f"Unimodal Audio loss: {audio_loss}")

            # print(f"Audioembeddings shape: {audio_embeddings.shape}")
            # print(f"Audio conv embeddings shape: {audio_conv_embeddings.shape}")

        text_embeddings = None
        text_mm_projection = None
        text_logits = None
        text_loss = None
        if text_inputs is not None:
            # print(f"Text (pot. masked) input: {text_inputs.input_ids}")
            text = self.tokenizer.batch_decode(text_inputs.input_ids)
            print(f"Text (pot. masked) input decoded: {text[0]}")
            if text_labels is not None:
                text_outputs = self.text_encoder(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask,
                    token_type_ids=text_inputs.token_type_ids,
                    labels=text_labels,
                    output_hidden_states=True,
                )
                text_loss = text_outputs.loss
                # print(f"Text labels: {text_labels}")
                print(f"Unimodal Text loss: {text_loss}")
                text_logits = text_outputs.logits
                flava_text_loss = self._get_flava_mlm_loss(
                    logits=text_logits, labels=text_labels
                )
                print(f"Flava Text loss: {flava_text_loss}")

            else:
                text_outputs = self.text_encoder(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask,
                    token_type_ids=text_inputs.token_type_ids,
                    output_hidden_states=True,
                )

            text_embeddings = text_outputs.hidden_states[-1]
            text_mm_projection = self.text_to_mm_projection(text_embeddings)
            text_logits = text_outputs.logits

            # probs = torch.nn.Softmax(dim=-1)(text_logits)
            # prob_of_labels = probs[text_labels][text_labels != -100]
            # print(f"Text probs: {probs.shape}")
            pred_tokens = torch.argmax(text_logits, dim=-1)
            pred_text = self.tokenizer.batch_decode(pred_tokens)
            print(f"Argmax text prediction: {pred_text[0]}")

        multimodal_embeddings = None
        if (
            audio_mm_projection is not None
            and text_mm_projection is not None
            and not skip_multimodal_encoder
        ):
            multimodal_embeddings = self.multimodal_encoder(
                torch.cat((audio_mm_projection, text_mm_projection), dim=1)
            )
            # print(f"Multimodal embeddings shape: {multimodal_embeddings.shape}")

        return MultimodalModelOutput(
            audio_embeddings=audio_embeddings,
            audio_conv_embeddings=audio_conv_embeddings,
            audio_loss=audio_loss,
            audio_mask=audio_mask,
            text_embeddings=text_embeddings,
            text_logits=text_logits,
            text_loss=text_loss,
            multimodal_embeddings=multimodal_embeddings,
        )

    def _mask_text_tokens(self, inputs, mask_rate):
        """
        inputs: input_ids, attention_mask, token_type_ids
        """
        labels = inputs.input_ids.clone()
        inputs_cloned = deepcopy(inputs).to(DEVICE)

        # print(f"Input ids[0]: {inputs.input_ids[0]}")

        # Probability matrix should only allow masking where attention_mask is 1
        if inputs.attention_mask is not None:
            # Use the attention_mask to limit where tokens can be masked
            probability_matrix = (
                torch.full(
                    labels.shape,
                    mask_rate,
                ).to(DEVICE)
                * inputs.attention_mask
            )
        else:
            # If no attention_mask is provided, tokens can be masked anywhere
            probability_matrix = torch.full(labels.shape, mask_rate).to(DEVICE)

        # Determine which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Mask tokens
        inputs_cloned.input_ids[masked_indices] = self.tokenizer.convert_tokens_to_ids(
            "[MASK]"
        )

        # Replace -100 in labels that we do not want to compute the loss for
        labels[~masked_indices] = -100

        return inputs_cloned, labels

    def _get_flava_mlm_loss(self, logits, labels):
        # manual FLAVA style MLM loss with manual ignoring index
        masked_tokens = labels.ne(-100)
        mlm_labels_filtered = labels[masked_tokens]
        logits_filtered = logits[masked_tokens, :]
        # print(f"MLM labels: {mlm_labels_filtered}")
        # argmax of logits_filtered along vocab dimension
        # print(f"MLM argmax tokens: {torch.argmax(logits_filtered, dim=-1)}")
        # print logits of the correct tokens
        logits_filtered_softmax = torch.nn.Softmax(dim=-1)(logits_filtered)
        # print(
        #     f"MLM logits: {logits_filtered_softmax[range(len(mlm_labels_filtered)), mlm_labels_filtered]}"
        # )

        loss = F.cross_entropy(
            logits_filtered.view(-1, self.text_encoder.config.vocab_size),
            mlm_labels_filtered.view(-1),
        )
        return loss

    def _global_contrastive_loss(
        self, audio_clf_tokens, text_clf_tokens, temperature=0.1
    ):
        # Linearly project into an embedding space
        audio_proj = self.audio_clf_projection(audio_clf_tokens)
        text_proj = self.text_clf_projection(text_clf_tokens)

        # Normalize the projections for stable training
        audio_proj = nn.functional.normalize(audio_proj, dim=-1)
        text_proj = nn.functional.normalize(text_proj, dim=-1)

        # Compute temperature-scaled logits
        logits_per_audio = (
            torch.matmul(audio_proj, text_proj.transpose(0, 1)) / temperature
        )
        logits_per_text = (
            torch.matmul(text_proj, audio_proj.transpose(0, 1)) / temperature
        )

        # print(f"GC logits per text: {logits_per_text}")
        # print(f"GC logits per audio: {logits_per_audio}")

        # Create labels
        labels = torch.arange(audio_proj.size(0), device=audio_proj.device)
        # labels are the same for both audio and text

        # Compute contrastive loss
        loss_audio = nn.functional.cross_entropy(logits_per_audio, labels)
        loss_text = nn.functional.cross_entropy(logits_per_text, labels)

        # Combine the two losses
        contrastive_loss = (loss_audio + loss_text) / 2
        # print(f"GC contrastive loss: {contrastive_loss}")

        return contrastive_loss

    def _audio_contrastive_loss(
        self, preds, labels, mask, temp=0.1, num_negatives=20, beta=0.1
    ):
        contrastive_loss = torch.tensor(0.0).to(DEVICE)

        if self.hparams.mm_audio_mask_rate > 0:
            contrastive_loss_fn = Wav2Vec2ContrastiveLoss(
                mask_rate=self.hparams.mm_audio_mask_rate,
                reduction="mean",
                temperature=temp,
                num_negatives=num_negatives,
            )
            # print(f"MMM audio preds shape {preds.shape}")
            # print(f"MMM audio labels shape {labels.shape}")
            # print(f"MMM audio mask shape {mask.shape}")
            contrastive_loss = contrastive_loss_fn(
                preds=preds,
                targets=labels,
                mask_time_indices=mask,
            )

        return contrastive_loss

    def _audio_text_matching_loss(
        self, audio_inputs, text_inputs, num_negatives=20, temp=0.1
    ):
        """
        audio_inputs: input_features (bs, seqlen, feat), attention_mask
        text_inputs: input_ids (bs, seqlen, feat), attention_mask, token_type_ids
        """

        # permute some of the batch and predict matching between audio and text
        bs = audio_inputs.input_features.shape[0]  # batch size
        permuted_audio_inputs = deepcopy(audio_inputs)
        permuted_text_inputs = deepcopy(text_inputs)

        # keep first half matched, second half permuted
        half_bs = bs // 2  # half of the batch size
        permutation_idx = torch.randperm(bs)  # generate permutation index
        permuted_audio_inputs.input_features[half_bs:] = audio_inputs.input_features[
            permutation_idx[half_bs:]
        ]
        permuted_text_inputs.input_ids[half_bs:] = text_inputs.input_ids[
            permutation_idx[half_bs:]
        ]
        permuted_text_inputs.attention_mask[half_bs:] = text_inputs.attention_mask[
            permutation_idx[half_bs:]
        ]
        permuted_text_inputs.token_type_ids[half_bs:] = text_inputs.token_type_ids[
            permutation_idx[half_bs:]
        ]

        # create labels, first half 0, second half 1
        labels = (
            torch.cat([torch.zeros(half_bs), torch.ones(bs - half_bs)])
            .to(audio_inputs.input_features.device)
            .long()
        )

        # mask text
        masked_permuted_text_inputs, _ = self._mask_text_tokens(
            inputs=permuted_text_inputs,
            mask_rate=self.hparams.mm_text_mask_rate,
        )

        # pass them through the multimodal encoder
        outputs = self.forward(
            audio_inputs=permuted_audio_inputs,
            text_inputs=masked_permuted_text_inputs,
            skip_multimodal_encoder=False,
        )

        multimodal_clf_tokens = outputs.multimodal_embeddings[:, 0, :]

        # classify matches
        logits = self.audio_text_matching_head(multimodal_clf_tokens)

        # compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return loss

    def _masked_mmm_loss(
        self,
        masked_multimodal_embeddings,
        audio_labels,
        audio_mask,
        text_labels,
    ):
        # Contrastive loss for audio
        audio_logits = self.audio_mm_mam_head(masked_multimodal_embeddings)
        # print(f"MMM audio logits shape {audio_logits.shape}")
        audio_loss = self._audio_contrastive_loss(
            preds=audio_logits[:, 1 : audio_labels.shape[1] + 1, :],
            labels=audio_labels,
            mask=audio_mask,
        )
        print(f"MMM contrastive audio loss {audio_loss}")

        # MLM text loss
        text_logits = self.text_mm_mlm_head(
            masked_multimodal_embeddings[:, audio_labels.shape[1] + 2 :, :]
        )

        # print(f"MMM text logits shape {text_logits.shape}")
        # print(f"MMM text labels shape {text_labels.shape}")

        loss_fct = nn.CrossEntropyLoss()  # ignore_index=-100
        text_loss = loss_fct(
            text_logits.view(-1, self.text_encoder.config.vocab_size),
            text_labels.view(-1),
        )
        print(f"MMM text loss {text_loss}")

        text_loss = self._get_flava_mlm_loss(text_logits, text_labels)
        print(f"MMM Flava text loss {text_loss}")

        return audio_loss, text_loss

    def _unimodal_step(self, audio_inputs, text_inputs, batch_idx, verbose=False):
        print(f"-----Unimodal step {batch_idx}")
        loss = 0.0
        audio_loss, text_loss = None, None

        if self.hparams.mam_weight > 0.0 and audio_inputs is not None:
            # masked_audio_inputs, audio_mask = self._mask_audio_tokens(audio_inputs)
            outputs = self.forward(audio_inputs=audio_inputs, output_audio_loss=True)
            audio_loss = outputs.audio_loss
            loss += self.hparams.mam_weight * audio_loss
            if verbose:
                print(f"Unimodal audio loss: {audio_loss}")

        if self.hparams.mlm_weight > 0.0 and text_inputs is not None:
            masked_text_inputs, labels = self._mask_text_tokens(
                inputs=text_inputs, mask_rate=self.hparams.uni_text_mask_rate
            )
            # print(f"Text inputs: {text_inputs}")
            # print(f"Masked text inputs: {masked_text_inputs}")
            # print(f"Text labels: {labels}")
            outputs = self.forward(text_inputs=masked_text_inputs, text_labels=labels)
            text_loss = outputs.text_loss
            loss += self.hparams.mlm_weight * text_loss
            if verbose:
                print(f"Unimodal text loss: {text_loss}")

        return loss, audio_loss, text_loss

    def _multimodal_step(self, audio_inputs, text_inputs, batch_idx, verbose=False):
        print(f"-----Multimodal step for batch {batch_idx}")

        loss = 0.0  # multimodal loss

        # print(f"Text shapes {text_inputs['input_ids'].shape}")

        # unmasked pass for global contrastive loss
        print(f"--Computing unmasked multimodal encodings")
        outputs = self.forward(
            audio_inputs=audio_inputs,
            text_inputs=text_inputs,
            skip_multimodal_encoder=True,
            output_audio_loss=False,
        )
        audio_embeddings = outputs.audio_embeddings
        text_embeddings = outputs.text_embeddings

        # print(f"Audio Shapes: {audio_embeddings.shape}")

        # mask audio and text
        # masked_audio_inputs, audio_mask_indices = self._mask_audio_tokens(audio_inputs)
        masked_text_inputs, text_labels = self._mask_text_tokens(
            inputs=text_inputs,
            mask_rate=self.hparams.mm_text_mask_rate,
        )
        # multimodal pass for other losses
        print(f"--Computing masked multimodal encodings")
        masked_outputs = self.forward(
            audio_inputs=audio_inputs,
            text_inputs=masked_text_inputs,
            skip_multimodal_encoder=False,
            output_audio_loss=True,
        )
        masked_audio_embeddings = masked_outputs.audio_embeddings
        masked_text_embeddings = masked_outputs.text_embeddings
        masked_multimodal_embeddings = masked_outputs.multimodal_embeddings
        audio_conv_embeddings = masked_outputs.audio_conv_embeddings
        audio_mask = masked_outputs.audio_mask

        # print(
        #     f"Masked multimodal embeddings after forward: {masked_multimodal_embeddings.shape}"
        # )

        # compute global contrastive loss
        gc_loss = None
        if self.hparams.global_contrastive_weight > 0:
            gc_loss = self._global_contrastive_loss(
                audio_clf_tokens=audio_embeddings[:, 0, :],
                text_clf_tokens=text_embeddings[:, 0, :],
            )
            self.train_gc_loss(gc_loss)
            loss += self.hparams.global_contrastive_weight * gc_loss
            print(f"GC loss: {gc_loss}")

        # compute masked mmm loss
        mm_audio_loss, mm_text_loss = None, None
        if self.hparams.mmm_audio_weight > 0 or self.hparams.mmm_text_weight > 0:
            mm_audio_loss, mm_text_loss = self._masked_mmm_loss(
                masked_multimodal_embeddings=masked_multimodal_embeddings,
                audio_labels=audio_conv_embeddings,
                audio_mask=audio_mask,
                text_labels=text_labels,
            )
            loss += self.hparams.mmm_audio_weight * mm_audio_loss
            loss += self.hparams.mmm_text_weight * mm_text_loss

        # text-audio matching loss
        audio_text_matching_loss = None
        if self.hparams.audio_text_matching_weight > 0:
            audio_text_matching_loss = self._audio_text_matching_loss(
                audio_inputs=audio_inputs,
                text_inputs=text_inputs,
            )
            loss += self.hparams.audio_text_matching_weight * audio_text_matching_loss

        return loss, gc_loss, mm_audio_loss, mm_text_loss, audio_text_matching_loss

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
            outputs = self.text_encoder(
                input_ids=token_sequences,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            predictions = outputs.logits
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
        audio_inputs, text_inputs = batch["audio_inputs"], batch["text_inputs"]
        mm_loss, uni_loss = 0.0, 0.0

        # if random.random() > self.hparams.unimodal_prob:
        # start_time = time.time()
        if self.hparams.unimodal_prob < 1.0:
            (
                mm_loss,
                gc_loss,
                mm_audio_loss,
                mm_text_loss,
                audio_text_matching_loss,
            ) = self._multimodal_step(audio_inputs, text_inputs, batch_idx)

            if gc_loss is not None:
                self.log(
                    "train/loss_gc", gc_loss, on_step=True, on_epoch=True, prog_bar=True
                )

            if mm_audio_loss is not None:
                self.log(
                    "train/loss_mm_mam",
                    mm_audio_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.train_mm_mam_loss(mm_audio_loss)

            if mm_text_loss is not None:
                self.log(
                    "train/loss_mm_mlm",
                    mm_text_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.train_mlm_text_loss(mm_text_loss)

            if audio_text_matching_loss is not None:
                self.log(
                    "train/loss_audio_text_matching",
                    audio_text_matching_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.train_audio_text_matching_loss(audio_text_matching_loss)

        # else:
        uni_loss, audio_loss, text_loss = self._unimodal_step(
            audio_inputs=audio_inputs, text_inputs=text_inputs, batch_idx=batch_idx
        )

        if audio_loss is not None:
            self.log(
                "train/loss_uni_mam",
                audio_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.train_uni_mam_loss(audio_loss)

        if text_loss is not None:
            self.log(
                "train/loss_uni_mlm",
                text_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.train_uni_mlm_loss(text_loss)

        return mm_loss + uni_loss

    def validation_step(self, batch, batch_idx):
        audio_inputs, text_inputs = batch["audio_inputs"], batch["text_inputs"]
        mm_loss, uni_loss = 0.0, 0.0

        # if random.random() > self.hparams.unimodal_prob:
        if self.hparams.unimodal_prob < 1.0:
            (
                mm_loss,
                gc_loss,
                mm_audio_loss,
                mm_text_loss,
                audio_text_matching_loss,
            ) = self._multimodal_step(audio_inputs, text_inputs, batch_idx)

            if gc_loss is not None:
                self.log(
                    "val/loss_gc", gc_loss, on_step=True, on_epoch=True, prog_bar=True
                )
                self.val_gc_loss(gc_loss)

            if mm_audio_loss is not None:
                self.log(
                    "val/loss_mm_mae",
                    mm_audio_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.val_mm_mam_loss(mm_audio_loss)

            if mm_text_loss is not None:
                self.log(
                    "val/loss_mm_mlm",
                    mm_text_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.val_mlm_text_loss(mm_text_loss)

            if audio_text_matching_loss is not None:
                self.log(
                    "val/loss_audio_text_matching",
                    audio_text_matching_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.val_audio_text_matching_loss(audio_text_matching_loss)

        # else:
        uni_loss, audio_loss, text_loss = self._unimodal_step(
            audio_inputs=audio_inputs, text_inputs=text_inputs, batch_idx=batch_idx
        )

        if audio_loss is not None:
            self.log(
                "val/loss_uni_mam",
                audio_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.val_uni_mam_loss(audio_loss)

        if text_loss is not None:
            self.log(
                "val/loss_uni_mlm",
                text_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.val_uni_mlm_loss(text_loss)

            # text loss is the early stopping loss
            self.log("val/loss", text_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.val_loss(text_loss)

        return mm_loss + uni_loss

    def test_step(self, batch, batch_idx):
        audio_inputs, text_inputs = batch["audio_inputs"], batch["text_inputs"]
        mm_loss, uni_loss = 0.0, 0.0

        # if random.random() > self.hparams.unimodal_prob:
        if self.hparams.unimodal_prob < 1.0:
            (
                mm_loss,
                gc_loss,
                mm_audio_loss,
                mm_text_loss,
                audio_text_matching_loss,
            ) = self._multimodal_step(audio_inputs, text_inputs, batch_idx)

            if gc_loss is not None:
                self.log(
                    "test/loss_gc", gc_loss, on_step=True, on_epoch=True, prog_bar=True
                )
                self.test_gc_loss(gc_loss)

            if mm_audio_loss is not None:
                self.log(
                    "test/loss_mm_mam",
                    mm_audio_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.test_mm_mam_loss(mm_audio_loss)

            if mm_text_loss is not None:
                self.log(
                    "test/loss_mm_mlm",
                    mm_text_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.test_mm_mlm_loss(mm_text_loss)

            if audio_text_matching_loss is not None:
                self.log(
                    "test/loss_audio_text_matching",
                    audio_text_matching_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.test_audio_text_matching_loss(audio_text_matching_loss)

        # else:
        uni_loss, audio_loss, text_loss = self._unimodal_step(
            audio_inputs=audio_inputs, text_inputs=text_inputs, batch_idx=batch_idx
        )

        if audio_loss is not None:
            self.log(
                "test/loss_uni_mam",
                audio_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.test_uni_mam_loss(audio_loss)

        if text_loss is not None:
            self.log(
                "test/loss_uni_mlm",
                text_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.test_uni_mlm_loss(text_loss)

        # additionally evaluate pseudo log likelihood
        if batch_idx % 500 == 0:
            pll = self.compute_pll(text_inputs)
            self.log("test/pll", pll, on_step=True, on_epoch=True, prog_bar=True)
            self.test_pll(pll)

        return mm_loss + uni_loss

    def on_validation_epoch_end(self):
        # update val loss best
        self.val_loss_best(self.val_loss.compute())

    def on_epoch_end(self):
        # metrics are logged and reset automatically by lightning
        pass

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
