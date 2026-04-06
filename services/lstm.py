import torch
import torch.nn as nn
from services.rnn_base import BaseRNNAlgorithm

class _LSTMModel(nn.Module):
    """Pure LSTM layer."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super(_LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LstmAlgorithm(BaseRNNAlgorithm):
    """LSTM-based stock price prediction algorithm."""
    
    def __init__(self, lookback: int = 10, hidden_size: int = 64, num_layers: int = 1,
                epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001,
                patience: int = 5, random_state: int = 42) -> None:
        super().__init__(
            model_type='lstm',
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
        """Build LSTM model."""
        return _LSTMModel(input_size, self.hidden_size, self.num_layers)

    def name(self) -> str:
        return "lstm"

    def feature_profile(self) -> str:
        return "lstm"
