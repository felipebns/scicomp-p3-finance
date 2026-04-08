import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import json

class Backtest:
    """Backtesting engine to compare trading strategies"""
    
    def __init__(self, test_df: pd.DataFrame, initial_capital: float = 10000):
        self.test_df = test_df.copy()
        self.initial_capital = initial_capital
        self.strategies = {}
        
    def run_all_strategies(self, model_predictions: np.ndarray) -> Dict[str, Dict]:
        """Run all benchmark strategies and model strategy"""
        results = {}
        
        # Model Strategy
        results["ML Model"] = self._model_strategy(model_predictions)
        
        # Buy and Hold
        results["Buy & Hold"] = self._buy_and_hold()
        
        # Always Long (100% always)
        results["Always Long"] = self._always_long()
        
        # Mean Reversion Strategy
        results["Mean Reversion"] = self._mean_reversion()
        
        # Random Walk
        results["Random Walk"] = self._random_walk()
        
        return results
    
    def _model_strategy(self, predictions: np.ndarray) -> Dict:
        """Strategy based on ML model predictions"""
        returns = self.test_df["return"].values
        
        # If prediction is 1 (up): stay invested, else cash
        strategy_returns = np.where(predictions == 1, returns, 0)
        
        return self._calculate_metrics(strategy_returns)
    
    def _buy_and_hold(self) -> Dict:
        """Buy on day 1 and hold until the end"""
        returns = self.test_df["return"].values
        strategy_returns = returns  # Hold entire period
        
        return self._calculate_metrics(strategy_returns)
    
    def _always_long(self) -> Dict:
        """Always 100% invested (same as buy and hold for long-only)"""
        returns = self.test_df["return"].values
        strategy_returns = returns
        
        return self._calculate_metrics(strategy_returns)
    
    def _mean_reversion(self) -> Dict:
        """Buy when price < 20-day SMA, sell when price > SMA"""
        prices = self.test_df["close"].values
        returns = self.test_df["return"].values
        
        # Calculate 20-day SMA
        sma_20 = self.test_df["close"].rolling(window=20).mean().values
        
        # Signal: 1 if price < SMA (buy), 0 otherwise (stay in cash)
        signals = (prices < sma_20).astype(int)
        
        strategy_returns = np.where(signals == 1, returns, 0)
        
        return self._calculate_metrics(strategy_returns)
    
    def _random_walk(self) -> Dict:
        """Random 50/50 buy or sell signals"""
        np.random.seed(42)  # For reproducibility
        returns = self.test_df["return"].values
        
        # Random signals: 50% chance of being long
        random_signals = np.random.randint(0, 2, size=len(returns))
        
        strategy_returns = np.where(random_signals == 1, returns, 0)
        
        return self._calculate_metrics(strategy_returns)
    
    def _calculate_metrics(self, strategy_returns: np.ndarray) -> Dict:
        """Calculate performance metrics for a strategy"""
        
        # Cumulative returns
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1
        total_return = cumulative_returns[-1]
        
        # Daily equity curve
        equity_curve = self.initial_capital * (1 + cumulative_returns)
        
        # Sharpe Ratio (annualized)
        if np.std(strategy_returns) > 0:
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max Drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win Rate
        win_rate = np.sum(strategy_returns > 0) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        # Cumulative return
        cum_ret = equity_curve[-1] - self.initial_capital
        
        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "final_equity": float(equity_curve[-1]),
            "equity_curve": equity_curve,
            "daily_returns": strategy_returns
        }
    
    def plot_backtest_results(self, backtest_results: Dict, output_dir: str) -> None:
        """Plot backtest results comparison"""
        
        # Prepare data for plotting
        dates = self.test_df["date"].values
        equity_curves = {name: data["equity_curve"] for name, data in backtest_results.items()}
        
        # Plot 1: Equity Curves
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Equity curves over time
        for strategy_name, equity_curve in equity_curves.items():
            ax1.plot(dates, equity_curve, label=strategy_name, linewidth=2, alpha=0.8)
        
        ax1.set_title("Backtest: Equity Curve Comparison (Last 10 Years)", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Equity ($)")
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.initial_capital, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
        
        # Plot 2: Performance Metrics
        metrics_names = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
        metrics_data = {strategy: [] for strategy in backtest_results.keys()}
        
        for metric_name in metrics_names:
            for strategy_name, data in backtest_results.items():
                metrics_data[strategy_name].append(data[metric_name])
        
        # Create a summary table
        metrics_df = pd.DataFrame(metrics_data, index=metrics_names).T
        
        # Plot metrics as bars
        x = np.arange(len(metrics_df.index))
        width = 0.15
        
        for i, col in enumerate(metrics_df.columns):
            offset = width * (i - len(metrics_df.columns) / 2)
            ax2.bar(x + offset, metrics_df[col], width, label=col, alpha=0.8)
        
        ax2.set_xlabel("Strategy")
        ax2.set_ylabel("Score")
        ax2.set_title("Performance Metrics Comparison", fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_df.index, rotation=45, ha='right')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/backtest_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Drawdown Analysis
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for strategy_name, equity_curve in equity_curves.items():
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max * 100
            ax.plot(dates, drawdown, label=strategy_name, linewidth=2, alpha=0.8)
        
        ax.set_title("Drawdown Analysis Over Time", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.fill_between(dates, 0, -100, alpha=0.1, color='red')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/drawdown_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_backtest_summary(self, backtest_results: Dict, output_dir: str) -> None:
        """Save backtest results to JSON"""
        
        summary = {}
        for strategy_name, metrics in backtest_results.items():
            summary[strategy_name] = {
                "total_return": float(metrics["total_return"]),
                "sharpe_ratio": float(metrics["sharpe_ratio"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "win_rate": float(metrics["win_rate"]),
                "final_equity": float(metrics["final_equity"])
            }
        
        with open(f"{output_dir}/backtest_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary to console
        print("\n" + "="*70)
        print("BACKTEST SUMMARY - STRATEGY COMPARISON")
        print("="*70)
        for strategy_name, metrics in summary.items():
            print(f"\n{strategy_name}:")
            print(f"  Total Return: {metrics['total_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
            print(f"  Final Equity: ${metrics['final_equity']:.2f}")
        print("="*70 + "\n")
