"""
src/models/rnn_model.py
========================
Vanilla RNN for market direction classification.

Architecture:
  Input  -> nn.RNN (num_layers, hidden_size, dropout) -> last hidden state
          -> nn.Linear(hidden_size, 1) -> sigmoid output

Vanilla RNN is the baseline: it has no gating mechanism so it struggles
with long-range dependencies. We include it to demonstrate empirically
why LSTM and GRU outperform it on financial sequences.

All three model classes (VanillaRNN, LSTMModel, GRUModel) expose the
same interface so the Trainer can swap between them without changes.
"""

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    """
    Two-layer Vanilla RNN followed by a linear classifier head.

    Parameters
    ----------
    input_size  : number of input features per time step
    hidden_size : number of RNN hidden units per layer
    num_layers  : number of stacked RNN layers
    dropout     : dropout probability between RNN layers (0 = disabled)
    bidirectional: if True, doubles effective hidden size
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int  = 2,
        dropout: float   = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Core RNN -- dropout only applies between layers (not after the last)
        self.rnn = nn.RNN(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,         # input shape: (batch, seq, features)
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )

        # Dropout before the classifier head (regularisation)
        self.dropout = nn.Dropout(p=dropout)

        # Linear projection: hidden -> 1 logit (BCEWithLogitsLoss handles sigmoid)
        self.classifier = nn.Linear(hidden_size * self.num_directions, 1)

        self._init_weights()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (batch_size, seq_len, input_size)

        Returns
        -------
        logits : (batch_size,) -- raw logit before sigmoid
        """
        # out: (batch, seq_len, hidden * directions)
        # h_n: (num_layers * directions, batch, hidden)
        out, h_n = self.rnn(x)

        # Take the final time-step output (last hidden state)
        # out[:, -1, :] is equivalent to h_n[-1] for single-direction
        last_hidden = out[:, -1, :]           # (batch, hidden * directions)
        last_hidden = self.dropout(last_hidden)
        logits      = self.classifier(last_hidden).squeeze(-1)  # (batch,)
        return logits

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Xavier uniform initialisation for weight matrices."""
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return "VanillaRNN"
