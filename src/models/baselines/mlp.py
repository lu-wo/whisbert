import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_p):
        super(MLP, self).__init__()
        self.layers = self._build_layers(
            input_size, hidden_size, output_size, num_layers, dropout_p
        )

    def _build_layers(
        self, input_size, hidden_size, output_size, num_layers, dropout_p
    ):
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_p))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(hidden_size, output_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
