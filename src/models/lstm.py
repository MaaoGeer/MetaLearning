"""Single-direction LSTM classifier for few-shot NIDS adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn

from .recurrent import CustomLSTM


class LSTMClassifier(nn.Module):
    """Window -> single-direction LSTM -> last hidden state -> linear classifier."""

    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.lstm = CustomLSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_size, n_classes)
        self._init_classifier()

    def _init_classifier(self) -> None:
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        h = self.lstm(x)
        last = h[:, -1, :]
        logits = self.classifier(last)
        if return_embedding:
            return logits, last
        return logits
