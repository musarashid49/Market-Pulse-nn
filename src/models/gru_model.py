"""
src/models/gru_model.py
========================
GRU (Gated Recurrent Unit) for market direction classification.

Architecture:
  Input -> nn.GRU (num_layers, hidden_size, dropout) -> last hidden state
         -> nn.Linear(hidden_size, 1) -> logit

GRU simplifies LSTM by merging the cell state and hidden state and using
only two gates (reset, update) vs LSTM's three. In practice GRU often
matches LSTM accuracy with fewer parameters and faster training.
This makes it the preferred choice when compute is the constraint.

Interface is identical to VanillaRNN and LSTMModel.
"""

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    Two-layer GRU with dropout and a linear classifier head.

    Parameters
    ----------
    input_size    : number of features per time step
    hidden_size   : GRU hidden units per layer
    num_layers    : number of stacked GRU layers
    dropout       : dropout between layers and before the head
    bidirectional : if True, runs GRU in both directions (BiGRU)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int  = 128,
        num_layers: int   = 2,
        dropout: float    = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.bidirectional  = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )

        self.dropout    = nn.Dropout(p=dropout)
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
        logits : (batch_size,)
        """
        # out: (batch, seq_len, hidden * directions)
        # h_n: (num_layers * directions, batch, hidden)
        out, h_n = self.gru(x)

        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        logits      = self.classifier(last_hidden).squeeze(-1)
        return logits

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return "GRU"
