import math
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoderLayer,
    WhisperEncoder,
    WhisperConfig,
    WhisperPreTrainedModel,
)

from src.utils.hf_utils import MyWhisperOutput
from src.utils.torch_utils import Wav2Vec2ContrastiveLoss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class GeneralEncoder(nn.Module):
    def __init__(self, feature_dim=20, intermediate_dim=768, nhead=8, num_layers=6):
        super(GeneralEncoder, self).__init__()

        # Map from input dimension to intermediate dimension
        self.mlp = nn.Linear(feature_dim, intermediate_dim)

        # Positional encoding layer
        self.pos_encoder = PositionalEncoding(intermediate_dim)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            intermediate_dim, nhead, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, intermediate_dim))

    def forward(self, x, mask=None):
        # print(f"MM Encoder input shape: {x.shape}")
        # Map from input dimension to intermediate dimension
        x = self.mlp(x)

        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1]).to(x.device)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Add classification token
        x = torch.cat((self.cls_token.repeat(x.shape[0], 1, 1), x), dim=1).to(x.device)

        # print(f"Shape of x after cat: {x.shape}")

        # Transpose mask, add mask for classification token, and transpose back
        mask = torch.cat(
            (torch.ones(mask.shape[0], 1).to(mask.device), mask), dim=1
        ).to(x.device)

        # Pass through transformer encoder
        # print(f"Shape of x: {x.shape}")
        # print(f"Shape of mask: {mask.shape}")
        output = self.transformer_encoder(x, src_key_padding_mask=mask)

        return output


class MyWhisperEncoder(WhisperEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.mask_rate = config.mask_rate
        self.mask_value = config.mask_value
        self.loss_name = config.loss_name
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.contrastive_loss_fn = Wav2Vec2ContrastiveLoss(
            mask_rate=self.mask_rate,
            reduction="mean",
            temperature=0.1,
            num_negatives=20,
            mask_length=1,
        )

        self.conv1 = nn.Conv1d(
            self.num_mel_bins, embed_dim, kernel_size=config.kernel_size, padding=1
        )
        self.conv2 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=1,
        )

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

        # LUKAS: Add a learnable classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.d_model))

        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        output_loss: Optional[bool] = True,
        mask_rate=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        mask_rate = mask_rate if mask_rate is not None else self.mask_rate

        print(f"input_features {input_features.shape}")

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        conv_embeds = inputs_embeds.clone()

        # mask inputs
        mask = None
        if output_loss:
            mask = self.contrastive_loss_fn.create_mask(
                (inputs_embeds.shape[0], inputs_embeds.shape[1])
            )
            inputs_embeds, mask = self._mask_audio_patches(x=inputs_embeds, mask=mask)

            # print(f"inputs_embeds {inputs_embeds.shape}")
            # print(f"conv_embeds {conv_embeds.shape}")
            # print(f"mask {mask.shape}")

        embed_pos = self.embed_positions.weight

        # LUKAS: adapt embed_pos to any size:
        embed_pos = embed_pos[: inputs_embeds.size(1), :].unsqueeze(0)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # print("hidden_states", hidden_states.shape)

        cls_tokens = self.cls_token.expand(
            input_features.size(0), -1, -1
        )  # (1, batch_size, d_model)
        # #print("cls_tokens", cls_tokens.shape)
        hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)

        # print("hidden_states", hidden_states.shape)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (
                dropout_probability < self.layerdrop
            ):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # print(f"hidden_states {hidden_states.shape}")

        loss = None
        if output_loss:
            if self.loss_name == "contrastive":
                loss = self.contrastive_loss_fn(
                    preds=hidden_states[:, 1:, :],
                    targets=conv_embeds,
                    mask_time_indices=mask,
                )
            elif self.loss_name == "l2":
                raise NotImplementedError
                # loss = self.l2_loss(hidden_states[:, 1:, :], conv_embeds, mask)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return MyWhisperOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            conv_embeddings=conv_embeds,
            loss=loss,
            mask=mask,
        )

    def _mask_audio_patches(self, x: torch.Tensor, mask: np.array, mask_value=None):
        """
        x: input after convolutional shape of shape (bs, seqlen, feat)
        return: x_masked, mask (bs, seqlen)
        """
        mask_value = mask_value if mask_value is not None else self.mask_value
        # print(f"shapes of x and mask: {x.shape}, {mask.shape}")
        torch_mask = torch.from_numpy(mask)
        x_masked = x.clone()
        x_masked[
            torch_mask.unsqueeze(-1).expand_as(x_masked)
        ] = mask_value  # Apply the mask across all dimensions
        return x_masked.to(x.device), mask

    # def l2_loss(self, preds, labels, mask):
    #     """
    #     preds: predictions/reconstructions of shape (bs, seqlen, feat)
    #     labels: labels of shape (bs, seqlen, feat)
    #     mask: mask of shape (bs, seqlen)
    #     """
    #     loss = ((preds - labels) ** 2).sum(dim=-1)
    #     loss = (loss * mask).sum() / mask.sum()
    #     return loss

    def load_weights(self, pretrain_model: WhisperEncoder):
        # Load conv1 and conv2 layers
        self.conv1.load_state_dict(pretrain_model.conv1.state_dict())
        self.conv2.load_state_dict(pretrain_model.conv2.state_dict())

        # Load embed_positions layer
        # self.embed_positions.load_state_dict(
        #     pretrain_model.embed_positions.state_dict()
        # )

        # Load layers from the WhisperEncoderLayer
        for my_layer, pretrain_layer in zip(self.layers, pretrain_model.layers):
            my_layer.load_state_dict(pretrain_layer.state_dict())

        # Load layer_norm
        self.layer_norm.load_state_dict(pretrain_model.layer_norm.state_dict())
