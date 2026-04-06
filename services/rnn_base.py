import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from abc import abstractmethod
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from services.algorithm import Algorithm


class BaseRNNAlgorithm(Algorithm):
    """Base class for RNN-based algorithms (LSTM, GRU)."""

    def __init__(self, model_type: str, lookback: int = 10, hidden_size: int = 64, num_layers: int = 1,
                epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001,
                patience: int = 5, random_state: int = 42) -> None:
        self.model_type = model_type  # 'lstm' or 'gru'
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.random_state = random_state
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        # Force CPU usage to avoid compatibility issues with older GPUs
        self.device = torch.device("cpu")
        torch.manual_seed(random_state)

    @abstractmethod
    def _build_model(self, input_size: int) -> nn.Module:
        """Build the RNN model. Subclasses must implement this."""
        raise NotImplementedError

    def _create_windows(self, X, y=None):
        n = len(X)
        X_windows = np.array([X[i - self.lookback : i] for i in range(self.lookback, n)])
        
        if y is None:
            return X_windows, None
        
        # y is indexed the same way to ensure pairing
        y_windows = np.array([y[i] for i in range(self.lookback, n)])
        return X_windows, y_windows

    def _prepare_data(self, df: pd.DataFrame, features: list[str], target_col: str = None, fit_scaler: bool = False):
        """Prepare and scale data, create windows."""
        X = df[features].values
        X = self.scaler_X.fit_transform(X) if fit_scaler else self.scaler_X.transform(X)
        
        if target_col is None:
            # Only X needed for prediction
            X_seq, _ = self._create_windows(X)
            X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            return X_tensor
        
        # Prepare target
        y = df[[target_col]].values.flatten()
        y = self.scaler_y.fit_transform(y.reshape(-1, 1)) if fit_scaler else self.scaler_y.transform(y.reshape(-1, 1))
        y = y.flatten()
        
        # Create paired windows in single call
        X_seq, y_seq = self._create_windows(X, y)
        
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1).to(self.device)
        
        return TensorDataset(X_tensor, y_tensor)

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str], target_col: str) -> None:
        """Train the RNN model with early stopping."""
        train_dataset = self._prepare_data(train_df, features, target_col, fit_scaler=True)
        
        if len(train_dataset) == 0:
            raise ValueError(f"Não há dados suficientes para criar janelas de tamanho {self.lookback}.")
        
        # Prepare validation data
        valid_dataset = self._prepare_data(valid_df, features, target_col, fit_scaler=False)
        if len(valid_dataset) == 0:
            raise ValueError(f"Validation set has insufficient data for windows of size {self.lookback}.")
            
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        
        input_size = len(features)
        self.model = self._build_model(input_size).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Early stopping parameters
        best_valid_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(self.epochs):
            # Training phase
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    valid_loss += loss.item()
            
            valid_loss /= len(valid_loader)
            self.model.train()
            
            # Early stopping check
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"[{self.model_type.upper()}] Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}")
            
            if patience_counter >= self.patience:
                print(f"[{self.model_type.upper()}] Early stopping at epoch {epoch+1} (no improvement for {self.patience} epochs)")
                break

    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        """Generate predictions."""
        if len(df) <= self.lookback:
            return np.full(len(df), np.nan)
            
        self.model.eval()
        X_tensor = self._prepare_data(df, features, fit_scaler=False)
        
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
            
        preds = self.scaler_y.inverse_transform(preds).flatten()
        
        padded_preds = np.full(len(df), np.nan)
        padded_preds[self.lookback:] = preds
        
        return padded_preds

    def name(self) -> str:
        return self.model_type

    def feature_profile(self) -> str:
        return self.model_type
