import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from services.transform import FeatureEngineer
from services.algorithms.base import Algorithm
from services.stock import Stock

class Pipeline:
    def __init__(
        self,
        stock: Stock,
        algorithms: list[Algorithm],
        output_dir: str = "output",
        test_size: float = 0.20,
        history_window: int = 250,
    ) -> None:
        self.stock = stock
        self.algorithms = algorithms
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = test_size
        self.history_window = history_window
        self.features = FeatureEngineer()

    def run(self, save: bool = True) -> dict[str, Any]:
        raw_df = self.stock.fetch()
        
        dataset, feature_cols, target_col = self.features.build(
            raw_df, self.algorithms[0].feature_profile()
        )

        train_df, test_df = self._split(dataset)
        
        results = {}
        for algo in self.algorithms:
            print(f"Training {algo.name()}...")
            algo.fit(train_df, pd.DataFrame(), feature_cols, target_col)
            metrics = self._evaluate(algo, test_df, feature_cols, target_col)
            results[algo.name()] = metrics

        if save:
            self._save_json(results)
            self._plot_metrics_comparison(results)
            
        return results

    def _split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        n = len(df)
        train_end = int(n * (1 - self.test_size))
        return (
            df.iloc[:train_end].copy(),
            df.iloc[train_end:].copy(),
        )

    def _evaluate(self, algorithm: Algorithm, df: pd.DataFrame, features: list[str], target_col: str) -> dict[str, float]:
        y_true = df[target_col].to_numpy()
        y_pred = algorithm.predict(df, features)
        y_prob = algorithm.predict_proba(df, features)
        actual_returns = df["return"].to_numpy()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5

        # Sharpe Ratio: Simular retornos com base nas predições
        # Se prediz 1 (up): compra, ganha o retorno real
        # Se prediz 0 (down): fica em cash, retorno = 0
        strategy_returns = np.where(y_pred == 1, actual_returns, 0)
        
        if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
            # Sharpe Ratio = média dos retornos / desvio padrão (com risk-free rate = 0)
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)  # Anualizando
        else:
            sharpe = 0.0

        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "auc": float(auc),
            "sharpe_ratio": float(sharpe)
        }
        print(f"[{algorithm.name()}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Sharpe: {sharpe:.4f}")
        return metrics

    def _save_json(self, metrics: dict[str, Any]) -> None:
        (self.output_dir / "classification_metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

    def _plot_metrics_comparison(self, results: dict[str, dict[str, float]]) -> None:
        df = pd.DataFrame(results).T
        fig, ax = plt.subplots(figsize=(12, 6))
        df.plot(kind="bar", ax=ax, width=0.8, alpha=0.9)
        ax.set_title("Model Metrics Comparison")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.5, 1.0)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
