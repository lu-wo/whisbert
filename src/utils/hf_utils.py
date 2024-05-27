from typing import Optional, Tuple

import torch
from transformers.file_utils import ModelOutput


class MultimodalModelOutput(ModelOutput):
    """
    Base class for outputs of multimodal models.
    Args:
        audio_embeddings (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Audio embeddings.
        text_embeddings (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Text embeddings.
        multimodal_embeddings (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Multimodal embeddings.
    """

    audio_embeddings: Optional[torch.FloatTensor] = None
    audio_conv_embeddings: Optional[torch.FloatTensor] = None
    audio_loss: Optional[torch.FloatTensor] = None
    audio_mask: Optional[torch.FloatTensor] = None
    text_embeddings: Optional[torch.FloatTensor] = None
    text_logits: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    multimodal_embeddings: Optional[torch.FloatTensor] = None


class MyWhisperOutput(ModelOutput):
    """
    Base class for model's Aoutputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        conv_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`): Convolutional
            embeddings of the input sequence after passing through the convolutional layers.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    conv_embeddings: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    mask: Optional[torch.FloatTensor] = None
