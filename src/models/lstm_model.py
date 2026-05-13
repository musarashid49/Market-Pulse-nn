"""
src/models/lstm_model.py
=========================
LSTM (Long Short-Term Memory) for market direction classification.

Architecture:
  Input -> nn.LSTM (num_layers, hidden_size, dropout) -> last hidden state
         -> nn.Linear(hidden_size, 1) -> logit

LSTM adds input, forget, and output gates to the vanilla RNN.
These gates allow the model to selectively remember and forget information
over long sequences -- critical for financial time-series where
month-old signals can still affect today's price.

Interface is identical to VanillaRNN and GRUModel.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Two-layer LSTM with dropout and a linear classifier head.

    Parameters
    ----------
    input_size    : number of features per time step
    hidden_size   : LSTM hidden units per layer
    num_layers    : number of stacked LSTM layers
    dropout       : dropout between layers and before the head
    bidirectional : if True, runs LSTM in both directions (BiLSTM)
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

        self.lstm = nn.LSTM(
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
        # c_n: (num_layers * directions, batch, hidden) -- cell state
        out, (h_n, c_n) = self.lstm(x)

        # Use the last time step's output
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
                nn.init.orthogonal_(param)   # orthogonal init for recurrent weights
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1.0 (helps remember by default)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return "LSTM"
