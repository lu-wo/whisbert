import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import ASTModel
from transformers.models.audio_spectrogram_transformer.configuration_audio_spectrogram_transformer import (
    ASTConfig,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import ModelOutput


class BaseModelOutputWithPoolingAndLoss(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states and a loss.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`return_loss=True` is passed or when the model is in training mode):
            The classification loss if the model has a classification head and was trained with labels.
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when :obj:`output_hidden_states=True` is passed or when :obj:`config.output_hidden_states=True`):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when :obj:`output_attentions=True` is passed or when :obj:`config.output_attentions=True`):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MaskedASTModel(ASTModel):
    def __init__(self, config: ASTConfig, mask_value: float = -100.0):
        super().__init__(config)
        self.mask_value = mask_value

    def mask_embeddings(
        self, embeddings: torch.Tensor, mask_rate: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a masked version of the embeddings and a mask tensor.

        The mask tensor has the same shape as the embeddings and contains
        True for each value that is masked and False elsewhere.
        """
        mask = torch.full(embeddings.shape[:2], False, dtype=torch.bool)
        num_mask = int(mask_rate * embeddings.shape[1])

        # We always mask the whole feat dimension, i.e. the whole token
        for i in range(embeddings.shape[0]):
            mask_indices = torch.randperm(embeddings.shape[1])[:num_mask]
            mask[i, mask_indices] = True

        masked_embeddings = embeddings.masked_fill(mask.unsqueeze(-1), self.mask_value)

        return masked_embeddings, mask

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mask_rate: float = 0.6,
        return_loss: bool = True,
        loss_fn: str = "l2",
    ):
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

        loss = None

        if input_values is None:
            raise ValueError("You have to specify input_values")

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(input_values)
        print(f"embedding_output.shape: {embedding_output.shape}")

        # Mask the embeddings
        masked_embedding_output, mask = self.mask_embeddings(
            embedding_output, mask_rate
        )
        print(f"masked_embedding_output.shape: {masked_embedding_output.shape}")
        print(f"mask shape {mask.shape}")

        encoder_outputs = self.encoder(
            masked_embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        pooled_output = (sequence_output[:, 0] + sequence_output[:, 1]) / 2

        # Compute the reconstruction loss on masked vectors only
        if return_loss:
            mask_expanded = mask.unsqueeze(-1).expand_as(sequence_output)
            masked_sequence_output = sequence_output[mask_expanded]
            masked_original = embedding_output[mask_expanded]
            print(f"masked_expanded.shape: {mask_expanded.shape}")
            print(f"masked_sequence_output.shape: {masked_sequence_output.shape}")
            print(f"masked_original.shape: {masked_original.shape}")

            if loss_fn == "l2":
                loss = F.mse_loss(masked_sequence_output, masked_original)
            elif loss_fn == "l1":
                loss = F.l1_loss(masked_sequence_output, masked_original)
            elif loss_fn == "cos_sim":
                loss = self.cos_sim_loss(sequence_output, masked_embedding_output, mask)
            else:
                raise ValueError(f"Loss function '{loss_fn}' not implemented")

        print(f"reconstruction_loss: {loss}")

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndLoss(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            loss=loss,
        )
