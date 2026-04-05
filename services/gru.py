import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from services.algorithm import Algorithm

class _GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, output_size: int = 1):
        super(_GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class GruAlgorithm(Algorithm):
    def __init__(self, lookback: int = 10, random_state: int = 42, epochs: int = 50, batch_size: int = 32) -> None:
        self.lookback = lookback
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        # Force CPU usage to avoid compatibility issues with older GPUs
        self.device = torch.device("cpu")
        torch.manual_seed(random_state)

    def _create_windows(self, X, y=None):
        X_windows, y_windows = [], []
        for i in range(self.lookback, len(X)):
            X_windows.append(X[i - self.lookback : i])
            if y is not None:
                y_windows.append(y[i])
        return np.array(X_windows), (np.array(y_windows) if y is not None else None)

    def _prepare_data(self, df: pd.DataFrame, features: list[str], target_col: str = None, fit_scaler: bool = False):
        X = df[features].values
        
        if fit_scaler:
            X = self.scaler_X.fit_transform(X)
        else:
            X = self.scaler_X.transform(X)
            
        y = None
        if target_col is not None:
            y = df[[target_col]].values
            if fit_scaler:
                y = self.scaler_y.fit_transform(y)
            else:
                y = self.scaler_y.transform(y)

        X_seq, y_seq = self._create_windows(X, y)
        
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        
        if target_col is not None:
            y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(self.device)
            return TensorDataset(X_tensor, y_tensor)
            
        return X_tensor

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str], target_col: str) -> None:
        train_dataset = self._prepare_data(train_df, features, target_col, fit_scaler=True)
        
        if len(train_dataset) == 0:
            raise ValueError(f"Não há dados suficientes para criar janelas de tamanho {self.lookback}.")
            
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        input_size = len(features)
        self.model = _GRUModel(input_size=input_size).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
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
        return "gru"

    def feature_profile(self) -> str:
        return "gru"
