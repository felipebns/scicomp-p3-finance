from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from services.transform import FeatureEngineer
from services.algorithm import Algorithm
from services.stock import Stock

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

class Pipeline:
    def __init__(
        self,
        stock: Stock,
        algorithm: Algorithm,
        output_dir: str = "output",
        test_size: float = 0.15,
        valid_size: float = 0.15,
        history_window: int = 250,
    ) -> None:
        self.stock = stock
        self.algorithm = algorithm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = test_size
        self.valid_size = valid_size
        self.history_window = history_window
        self.features = FeatureEngineer()

    def run(self, save: bool = True) -> dict[str, Any]:
        raw_df = self.stock.fetch()
        dataset, feature_cols, target_col = self.features.build(
            raw_df, self.algorithm.feature_profile()
        )

        train_df, valid_df, test_df = self._split(dataset)
        self.algorithm.fit(train_df, valid_df, feature_cols, target_col)

        metrics = self._evaluate(test_df, feature_cols, target_col)
        if save:
            self._save_json(metrics)
            self._plot(test_df, feature_cols, target_col)
        return metrics

    def _split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        train_end = int(n * (1 - self.valid_size - self.test_size))
        valid_end = int(n * (1 - self.test_size))
        return (
            df.iloc[:train_end].copy(),
            df.iloc[train_end:valid_end].copy(),
            df.iloc[valid_end:].copy(),
        )

    def _evaluate(self, df: pd.DataFrame, features: list[str], target_col: str) -> dict[str, Any]:
        y_true = df[target_col].to_numpy()
        y_pred = self.algorithm.predict(df, features)

        mask = ~np.isnan(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        y_rw = np.zeros_like(y_true_clean, dtype=float)

        last_adj_close = float(df["adj_close"].iloc[-1])

        return {
            "ticker": self.stock.ticker,
            "algorithm": self.algorithm.name(),
            "test": {
                "mae": float(mean_absolute_error(y_true_clean, y_pred_clean)),
                "rmse": float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))),
                "r2": float(r2_score(y_true_clean, y_pred_clean)),
            },
            "random_walk": {
                "mae": float(mean_absolute_error(y_true_clean, y_rw)),
                "rmse": float(np.sqrt(mean_squared_error(y_true_clean, y_rw))),
                "r2": float(r2_score(y_true_clean, y_rw)),
            },
            "last_adj_close": last_adj_close,
            "predicted_next_return": float(y_pred[-1]),
            "predicted_next_adj_close": float(last_adj_close * (1.0 + y_pred[-1])),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def _save_json(self, metrics: dict[str, Any]) -> None:
        (self.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

    def _plot(self, df: pd.DataFrame, features: list[str], target_col: str) -> None:
        if len(df) < 2:
            return

        plot_df = df.copy().reset_index(drop=True)
        pred_return = self.algorithm.predict(plot_df, features)
        actual_return = plot_df[target_col].to_numpy()
        random_walk = np.zeros_like(pred_return)

        if len(plot_df) > self.history_window:
            plot_df = plot_df.iloc[-self.history_window:].copy()
            pred_return = pred_return[-self.history_window:]
            actual_return = actual_return[-self.history_window:]
            random_walk = random_walk[-self.history_window:]

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(plot_df["date"], actual_return, label="Actual return")
        ax.plot(plot_df["date"], pred_return, label=f"{self.algorithm.name()} prediction")
        ax.plot(plot_df["date"], random_walk, label="Random walk (0)")
        ax.set_title("Adjusted Return Prediction vs Random Walk")
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        ax.legend()
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(self.output_dir / "return_comparison.png", dpi=150)
        plt.close(fig)