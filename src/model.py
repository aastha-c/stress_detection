"""
LSTM Model
==========
Bidirectional LSTM with attention for stress detection.
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Simple additive attention over LSTM hidden states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        # lstm_output: (batch, seq_len, hidden_size)
        weights = torch.softmax(self.attn(lstm_output), dim=1)  # (batch, seq_len, 1)
        return (lstm_output * weights).sum(dim=1)               # (batch, hidden_size)


class StressLSTM(nn.Module):
    """
    Bidirectional LSTM with attention for binary stress classification.

    Parameters
    ----------
    input_size   : number of features per time step
    hidden_size  : LSTM hidden dimension
    num_layers   : stacked LSTM layers
    dropout      : dropout rate
    num_classes  : output classes (2 for binary)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = Attention(hidden_size * 2)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        logits : (batch, num_classes)
        """
        # Batch-norm across features (reshape needed)
        batch, seq_len, feat = x.shape
        x = x.reshape(batch * seq_len, feat)
        x = self.input_bn(x)
        x = x.reshape(batch, seq_len, feat)

        lstm_out, _ = self.lstm(x)           # (batch, seq_len, hidden*2)
        context = self.attention(lstm_out)    # (batch, hidden*2)
        context = self.dropout(context)
        return self.classifier(context)       # (batch, num_classes)
