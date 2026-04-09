import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from typing import Dict

class Backtest:
    """Backtesting engine to compare threshold-based and benchmark trading strategies with realistic assumptions"""
    
    def __init__(self, test_df: pd.DataFrame, initial_capital: float = 10000,
                 transaction_cost: float = 0.0005, slippage: float = 0.0005,
                 annual_rf_rate: float = 0.05):
        self.test_df = test_df.copy()
        self.initial_capital = initial_capital
        self.tc = transaction_cost
        self.slippage = slippage
        self.annual_rf_rate = annual_rf_rate
        self.daily_rf_rate = (1 + annual_rf_rate) ** (1/252) - 1
        
    def run_threshold_strategies(self, model_probabilities: np.ndarray) -> Dict[str, Dict]:
        """Run multiple threshold-based strategies and compare with benchmarks.
        
        Args:
            model_probabilities: Array of predicted probabilities from ML model [0, 1]
        
        Returns:
            Dictionary mapping strategy names to their performance metrics
        """
        results = {}
        
        # Test multiple probability thresholds for entering positions
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
        for threshold in thresholds:
            strategy_name = f"ML Threshold {threshold:.2f}"
            results[strategy_name] = self._threshold_strategy(model_probabilities, threshold)
        
        # Add benchmark strategies
        results["Buy & Hold"] = self._buy_and_hold()
        results[f"Fixed Income {self.annual_rf_rate:.2%} annual"] = self._fixed_income()
        results["Mean Reversion"] = self._mean_reversion()
        results["Random Walk (Median)"] = self._random_walk_monte_carlo(n_runs=50)
        
        return results
    
    def run_all_strategies(self, model_predictions: np.ndarray) -> Dict[str, Dict]:
        """Legacy method for backward compatibility (deprecated).
        Use run_threshold_strategies() instead."""
        # Convert binary predictions to probabilities (0 -> 0.0, 1 -> 1.0)
        probs = model_predictions.astype(float)
        return self._threshold_strategy_simple(probs)
        
    def _threshold_strategy(self, probabilities: np.ndarray, threshold: float = 0.50) -> Dict:
        """Strategy: Buy when probability > threshold, stay in cash otherwise.
        
        Args:
            probabilities: Predicted probabilities [0, 1]
            threshold: Entry threshold
        """
        positions = (probabilities > threshold).astype(int)
        net_returns = self._calculate_strategy_returns(positions)
        return self._calculate_metrics(net_returns, positions)
    
    def _threshold_strategy_simple(self, probabilities: np.ndarray) -> Dict:
        """Simple wrapper for backward compatibility."""
        positions = (probabilities >= 0.5).astype(int)
        net_returns = self._calculate_strategy_returns(positions)
        return {"ML Model": self._calculate_metrics(net_returns, positions)}
        
    def _calculate_strategy_returns(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate net returns given a vector of target positions for each day.
        - positions[t]: Position taken at the CLOSE of day t (1=long, 0=cash).
        - Return depends on price change from t to t+1.
        - Costs are deducted when position changes.
        """
        next_returns = self.test_df["return"].shift(-1).fillna(0).values
        
        gross_returns = positions * next_returns
        
        position_changes = np.diff(positions, prepend=0)
        costs = np.abs(position_changes) * (self.tc + self.slippage)
        
        # If not in position (0), earn risk-free rate
        cash_interest = np.where(positions == 0, self.daily_rf_rate, 0)
        
        net_returns = gross_returns - costs + cash_interest
        return net_returns
    
    def _buy_and_hold(self) -> Dict:
        """Buy on day 1 and hold until the end (No turnover = No recurring costs)"""
        positions = np.ones(len(self.test_df))
        net_returns = self._calculate_strategy_returns(positions)
        return self._calculate_metrics(net_returns, positions)
    
    def _fixed_income(self) -> Dict:
        """Remains 100% in cash earning a Constant Interest Rate (Risk-Free benchmark)"""
        positions = np.zeros(len(self.test_df))
        net_returns = self._calculate_strategy_returns(positions)
        return self._calculate_metrics(net_returns, positions)
    
    def _mean_reversion(self) -> Dict:
        """Buy when price < 20-day SMA, sell when price >= SMA"""
        prices = self.test_df["close"].values
        sma_20 = self.test_df["close"].rolling(window=20).mean().values
        
        positions = (prices < sma_20).astype(int)
        net_returns = self._calculate_strategy_returns(positions)
        return self._calculate_metrics(net_returns, positions)
    
    def _random_walk_monte_carlo(self, n_runs: int = 50) -> Dict:
        """
        Robust Random Walk: Execute multiple runs and return median-representative path.
        Returns the run whose total_return is CLOSEST TO MEDIAN (not mean).
        Includes metadata about all runs for transparency.
        """
        np.random.seed(42)
        all_metrics = []
        
        for run_idx in range(n_runs):
            positions = np.random.randint(0, 2, size=len(self.test_df))
            net_returns = self._calculate_strategy_returns(positions)
            metrics = self._calculate_metrics(net_returns, positions)
            all_metrics.append(metrics)
        
        # Find run closest to median return (robust center)
        returns = [m["total_return"] for m in all_metrics]
        median_return = np.median(returns)
        distances = np.abs(np.array(returns) - median_return)
        median_idx = np.argmin(distances)
        
        representative_metrics = all_metrics[median_idx]
        
        # Store run statistics for transparency
        representative_metrics["_metadata"] = {
            "n_runs": n_runs,
            "median_return": float(median_return),
            "representative_run_idx": int(median_idx),
            "mean_return_all_runs": float(np.mean(returns)),
            "std_return_all_runs": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns))
        }
        
        return representative_metrics
    
    def _calculate_metrics(self, strategy_returns: np.ndarray, positions: np.ndarray, override_equity=None) -> Dict:
        """Calculate realistic performance metrics for a strategy"""
        
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1
        total_return = cumulative_returns[-1]
        
        if override_equity is not None:
            equity_curve = override_equity
        else:
            equity_curve = self.initial_capital * (1 + cumulative_returns)
            
        # Annualized Return
        years = len(strategy_returns) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe Ratio
        if np.std(strategy_returns) > 0:
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
            
        # Max Drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calculate gross returns before costs
        next_returns = self.test_df["return"].shift(-1).fillna(0).values
        gross_returns = positions * next_returns
        
        is_active = (positions > 0)
        active_days = np.sum(is_active)
        if active_days > 0:
            active_hit_rate = np.sum(gross_returns[is_active] > 0) / active_days
        else:
            active_hit_rate = 0.0
            
        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "active_hit_rate": float(active_hit_rate),
            "final_equity": float(equity_curve[-1]),
            "equity_curve": equity_curve,
            "daily_returns": strategy_returns
        }
    
    def plot_backtest_results(self, backtest_results: Dict, output_dir: str) -> None:
        """Plot backtest results: equity curves, metrics comparison, and threshold analysis"""
        dates = self.test_df["date"].values
        equity_curves = {name: data["equity_curve"] for name, data in backtest_results.items()}
        
        # Separate threshold strategies from benchmarks for better visualization
        threshold_strats = {k: v for k, v in equity_curves.items() if "Threshold" in k}
        benchmark_strats = {k: v for k, v in equity_curves.items() if "Threshold" not in k}
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))
        
        # Plot 1: All strategies equity curves (normalized)
        for strategy_name, equity_curve in equity_curves.items():
            normalized_curve = equity_curve / self.initial_capital
            linestyle = '--' if "Threshold" in strategy_name else '-'
            ax1.plot(dates, normalized_curve, label=strategy_name, linewidth=2.5, alpha=0.8, linestyle=linestyle)
            
        ax1.set_title("Backtest: Normalized Equity Curve Comparison (Thresholds vs Benchmarks)", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Normalized Equity (Multiple of Initial Capital)")
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: Metrics comparison
        metrics_names = ["annualized_return", "sharpe_ratio", "max_drawdown", "active_hit_rate"]
        metrics_data = {strategy: [] for strategy in backtest_results.keys()}
        
        for metric_name in metrics_names:
            for strategy_name, data in backtest_results.items():
                metrics_data[strategy_name].append(data[metric_name])
                
        metrics_df = pd.DataFrame(metrics_data, index=metrics_names).T
        
        x = np.arange(len(metrics_df.index))
        width = 0.18
        
        for i, col in enumerate(metrics_df.columns):
            offset = width * (i - len(metrics_df.columns) / 2)
            ax2.bar(x + offset, metrics_df[col], width, label=col, alpha=0.8)
            
        ax2.set_xlabel("Strategy")
        ax2.set_ylabel("Score")
        ax2.set_title("Performance Metrics Comparison", fontsize=14, fontweight='bold')
        ax2.set_ylim(-0.5, 3.0)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_df.index, rotation=45, ha='right', fontsize=9)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/backtest_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Separate plot: Focus on thresholds only
        if threshold_strats:
            fig, ax = plt.subplots(figsize=(14, 7))
            for strategy_name, equity_curve in threshold_strats.items():
                normalized_curve = equity_curve / self.initial_capital
                ax.plot(dates, normalized_curve, label=strategy_name, linewidth=2.5, alpha=0.8, marker='o', markersize=2)
            
            ax.set_title("ML Model: Effect of Probability Threshold on Equity Curve", fontsize=14, fontweight='bold')
            ax.set_xlabel("Date")
            ax.set_ylabel("Normalized Equity")
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/threshold_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # Drawdown analysis
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for strategy_name, equity_curve in equity_curves.items():
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max * 100
            linestyle = '--' if "Threshold" in strategy_name else '-'
            ax.plot(dates, drawdown, label=strategy_name, linewidth=2, alpha=0.8, linestyle=linestyle)
            
        ax.set_title("Drawdown Analysis Over Time", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.fill_between(dates, 0, -100, alpha=0.1, color='red')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/drawdown_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_backtest_summary(self, backtest_results: Dict, output_dir: str) -> None:
        """Save and print backtest summary with detailed statistics."""
        
        summary = {}
        for strategy_name, metrics in backtest_results.items():
            base_metrics = {
                "total_return": float(metrics["total_return"]),
                "annualized_return": float(metrics["annualized_return"]),
                "sharpe_ratio": float(metrics["sharpe_ratio"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "active_hit_rate": float(metrics["active_hit_rate"]),
                "final_equity": float(metrics["final_equity"])
            }
            
            # Add random walk metadata if present
            if "_metadata" in metrics:
                base_metrics["_random_walk_metadata"] = metrics["_metadata"]
            
            summary[strategy_name] = base_metrics
            
        with open(f"{output_dir}/backtest_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("\n" + "="*90)
        print("BACKTEST SUMMARY - STRATEGY COMPARISON")
        print("="*90)
        
        # Separate and print threshold strategies
        threshold_results = {k: v for k, v in summary.items() if "Threshold" in k}
        if threshold_results:
            print("\nMULTIPLE PROBABILITY THRESHOLDS:")
            print("-" * 90)
            for strategy_name in sorted(threshold_results.keys()):
                metrics = threshold_results[strategy_name]
                print(f"\n{strategy_name}:")
                print(f"  Total Return:     {metrics['total_return']*100:>8.2f}%")
                print(f"  Annualized Ret:   {metrics['annualized_return']*100:>8.2f}%")
                print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>8.4f}")
                print(f"  Max Drawdown:     {metrics['max_drawdown']*100:>8.2f}%")
                print(f"  Active Hit Rate:  {metrics['active_hit_rate']*100:>8.2f}%")
                print(f"  Final Equity:     ${metrics['final_equity']:>12,.2f}")
        
        # Print benchmark strategies
        benchmark_results = {k: v for k, v in summary.items() if "Threshold" not in k}
        if benchmark_results:
            print("\n\nBENCHMARK STRATEGIES:")
            print("-" * 90)
            for strategy_name, metrics in benchmark_results.items():
                print(f"\n{strategy_name}:")
                print(f"  Total Return:     {metrics['total_return']*100:>8.2f}%")
                print(f"  Annualized Ret:   {metrics['annualized_return']*100:>8.2f}%")
                print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>8.4f}")
                print(f"  Max Drawdown:     {metrics['max_drawdown']*100:>8.2f}%")
                print(f"  Active Hit Rate:  {metrics['active_hit_rate']*100:>8.2f}%")
                print(f"  Final Equity:     ${metrics['final_equity']:>12,.2f}")
                
                # Show random walk metadata if available
                if "_random_walk_metadata" in metrics:
                    meta = metrics["_random_walk_metadata"]
                    print(f"  [Random Walk: {meta['n_runs']} runs]")
                    print(f"    Median:  {meta['median_return']*100:.2f}% | Mean: {meta['mean_return_all_runs']*100:.2f}% ± {meta['std_return_all_runs']*100:.2f}%")
        
        print("\n" + "="*90 + "\n")
