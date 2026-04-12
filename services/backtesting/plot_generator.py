import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict


class PlotGenerator:
    """Generates backtest visualization plots."""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
    
    def plot_all(self, backtest_results: Dict, test_df: pd.DataFrame, output_dir: str) -> None:
        """Generate all backtest plots."""
        dates = np.sort(test_df["date"].unique())
        
        self._plot_equity_curves(backtest_results, dates, output_dir)
        self._plot_metrics_comparison(backtest_results, output_dir)
        self._plot_drawdown_analysis(backtest_results, dates, output_dir)
    
    def _plot_equity_curves(self, results: Dict, dates: np.ndarray, output_dir: str) -> None:
        """Plot equity curves: thresholds vs benchmarks."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))
        
        equity_curves = {name: data["equity_curve"] for name, data in results.items()}
        
        # Plot 1: All strategies
        for strategy_name, equity_curve in equity_curves.items():
            normalized = equity_curve / self.initial_capital
            linestyle = '--' if "Threshold" in strategy_name else '-'
            ax1.plot(dates, normalized, label=strategy_name, linewidth=2.5, 
                    alpha=0.8, linestyle=linestyle)
        
        ax1.set_title("Backtest: Normalized Equity Curve Comparison", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Normalized Equity (Multiple of Initial Capital)")
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: Metrics comparison
        self._plot_metrics_bars(results, ax2)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/backtest_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self, results: Dict, output_dir: str) -> None:
        """Plot metrics comparison as bars."""
        metrics_names = ["annualized_return", "sharpe_ratio", "max_drawdown", "active_hit_rate"]
        metrics_data = {strategy: [results[strategy][m] for m in metrics_names] 
                       for strategy in results.keys()}
        
        metrics_df = pd.DataFrame(metrics_data, index=metrics_names).T
        
        x = np.arange(len(metrics_df.index))
        width = 0.18
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for i, col in enumerate(metrics_df.columns):
            offset = width * (i - len(metrics_df.columns) / 2)
            ax.bar(x + offset, metrics_df[col], width, label=col, alpha=0.8)
        
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Score")
        ax.set_title("Performance Metrics Comparison", fontsize=14, fontweight='bold')
        ax.set_ylim(-0.5, 3.0)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df.index, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_bars(self, results: Dict, ax) -> None:
        """Bar plot for metrics."""
        metrics_names = ["annualized_return", "sharpe_ratio", "max_drawdown", "active_hit_rate"]
        metrics_data = {strategy: [results[strategy][m] for m in metrics_names] 
                       for strategy in results.keys()}
        
        metrics_df = pd.DataFrame(metrics_data, index=metrics_names).T
        
        x = np.arange(len(metrics_df.index))
        width = 0.18
        
        for i, col in enumerate(metrics_df.columns):
            offset = width * (i - len(metrics_df.columns) / 2)
            ax.bar(x + offset, metrics_df[col], width, label=col, alpha=0.8)
        
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Score")
        ax.set_title("Performance Metrics Comparison", fontsize=14, fontweight='bold')
        ax.set_ylim(-0.5, 3.0)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df.index, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    def _plot_drawdown_analysis(self, results: Dict, dates: np.ndarray, output_dir: str) -> None:
        """Plot drawdown over time."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for strategy_name, data in results.items():
            equity_curve = data["equity_curve"]
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max * 100
            linestyle = '--' if "Threshold" in strategy_name else '-'
            ax.plot(dates, drawdown, label=strategy_name, linewidth=2, 
                   alpha=0.8, linestyle=linestyle)
        
        ax.set_title("Drawdown Analysis Over Time", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.fill_between(dates, 0, -100, alpha=0.1, color='red')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/drawdown_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
