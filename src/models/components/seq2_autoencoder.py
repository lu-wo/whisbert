import random

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        # concatenate the hidden states from the forward and backward pass
        hidden = torch.cat(
            (hidden[0 : hidden.size(0) : 2], hidden[1 : hidden.size(0) : 2]), 2
        )
        cell = torch.cat((cell[0 : cell.size(0) : 2], cell[1 : cell.size(0) : 2]), 2)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.fc(output)
        return output, (hidden, cell)


class Seq2SeqAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        output_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        teacher_forcing_ratio=0.5,
    ):
        super(Seq2SeqAutoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(input_size, hidden_size * 2, num_layers, output_size)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, source, target=None, debug=False):
        batch_size = source.size(0)
        seq_length = source.size(1)

        # Encoding
        hidden, cell = self.encoder(source)
        latent_space = hidden
        if debug:
            print(f"hidden shape {hidden.shape}, cell shape {cell.shape}")

        # Prepare space for the output of the decoder
        outputs = torch.zeros(batch_size, seq_length, 1).to(source.device)

        # Use last value of the input sequence as initial input for the decoder
        decoder_input = torch.zeros(batch_size, 1, 1).to(source.device)
        if debug:
            print(f"decoder input shape {decoder_input.shape}")

        # Decoding
        for t in range(seq_length):
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)

            # Decide whether to use teacher forcing
            teacher_force = (
                target is not None and random.random() < self.teacher_forcing_ratio
            )
            # If teacher forcing, use actual target as the next input. If not, use predicted output.
            decoder_input = target[:, t, :].unsqueeze(1) if teacher_force else output

        return outputs, latent_space
