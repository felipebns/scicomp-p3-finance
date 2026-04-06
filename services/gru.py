import torch
import torch.nn as nn
from services.rnn_base import BaseRNNAlgorithm

class _GRUModel(nn.Module):
    """Pure GRU layer."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super(_GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class GruAlgorithm(BaseRNNAlgorithm):
    """GRU-based stock price prediction algorithm."""

    def __init__(self, lookback: int = 10, hidden_size: int = 64, num_layers: int = 1,
                epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001,
                patience: int = 5, random_state: int = 42) -> None:
        super().__init__(
            model_type='gru',
            lookback=lookback,
            hidden_size=hidden_size,
            num_layers=num_layers,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            random_state=random_state
        )

    def _build_model(self, input_size: int) -> nn.Module:
        """Build GRU model."""
        return _GRUModel(input_size, self.hidden_size, self.num_layers)

    def name(self) -> str:
        return "gru"

    def feature_profile(self) -> str:
        return "gru"
