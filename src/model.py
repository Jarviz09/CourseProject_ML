"""Neural network model for time-series forecasting."""
from __future__ import annotations

import torch
from torch import nn


class EnergyLSTM(nn.Module):
    """LSTM-based regression model for one-step energy consumption forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a forecast for each input window."""
        lstm_output, _ = self.lstm(x)
        last_hidden = lstm_output[:, -1, :]
        return self.regressor(last_hidden)
