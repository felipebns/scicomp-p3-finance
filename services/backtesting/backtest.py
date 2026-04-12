import json
import pandas as pd
import numpy as np
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

from services.backtesting.metrics_calculator import MetricsCalculator
from services.backtesting.position_normalizer import PositionNormalizer
from services.backtesting.return_calculator import ReturnCalculator
from services.backtesting.plot_generator import PlotGenerator


class Backtest:
    """Backtesting engine for multi-asset trading strategies."""
    
    def __init__(self, test_df: pd.DataFrame, initial_capital: float = 10000,
                 transaction_cost: float = 0.0005, slippage: float = 0.0005,
                 annual_rf_rate: float = 0.05, position_sizing: str = "equal_weight"):
        self.test_df = test_df.copy()
        self.initial_capital = initial_capital
        self.annual_rf_rate = annual_rf_rate
        self.position_sizing = position_sizing
        
        # Initialize components
        self.position_normalizer = PositionNormalizer()
        self.return_calculator = ReturnCalculator(transaction_cost, slippage, annual_rf_rate)
        self.metrics_calculator = MetricsCalculator(initial_capital, annual_rf_rate)
        self.plot_generator = PlotGenerator(initial_capital)
    
    def run_threshold_strategies(self, model_probabilities: np.ndarray, 
                                 thresholds: list[float] = None) -> Dict[str, Dict]:
        """Run threshold-based strategies and benchmarks in parallel."""
        if thresholds is None:
            thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
        
        results = {}
        
        # Test probability thresholds in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            threshold_futures = {
                executor.submit(self._run_threshold_strategy, model_probabilities, th): th
                for th in thresholds
            }
            
            for future in threshold_futures:
                threshold = threshold_futures[future]
                results[f"ML Threshold {threshold:.2f}"] = future.result()
        
        # Add benchmarks (sequential, fast)
        results["Buy & Hold"] = self._run_buy_and_hold()
        results[f"Fixed Income {self.annual_rf_rate:.2%} annual"] = self._run_fixed_income()
        results["Mean Reversion"] = self._run_mean_reversion()
        results["Random Walk (Median)"] = self._run_random_walk()
        
        return results
    
    def _run_threshold_strategy(self, probabilities: np.ndarray, threshold: float) -> Dict:
        """Strategy: Buy when probability > threshold."""
        positions = (probabilities > threshold).astype(int)
        
        if self.position_sizing == "probability_weighted":
            positions = probabilities.copy()
            positions[probabilities <= threshold] = 0
            positions = self._apply_probability_weights(positions)
        
        positions = self.position_normalizer.normalize(positions, self.test_df)
        daily_returns = self.return_calculator.calculate(positions, self.test_df)
        return self.metrics_calculator.calculate(daily_returns, positions, self.test_df)
    
    def _run_buy_and_hold(self) -> Dict:
        """Buy on day 1 and hold."""
        positions = np.ones(len(self.test_df))
        positions = self.position_normalizer.normalize(positions, self.test_df)
        daily_returns = self.return_calculator.calculate(positions, self.test_df)
        return self.metrics_calculator.calculate(daily_returns, positions, self.test_df)
    
    def _run_fixed_income(self) -> Dict:
        """Stay 100% in cash."""
        positions = np.zeros(len(self.test_df))
        daily_returns = self.return_calculator.calculate(positions, self.test_df)
        return self.metrics_calculator.calculate(daily_returns, positions, self.test_df)
    
    def _run_mean_reversion(self) -> Dict:
        """Buy when price < 20-day SMA."""
        prices = self.test_df["close"].values
        sma_20 = self.test_df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(window=20).mean()
        ).bfill().values
        
        positions = (prices < sma_20).astype(int)
        positions = self.position_normalizer.normalize(positions, self.test_df)
        daily_returns = self.return_calculator.calculate(positions, self.test_df)
        return self.metrics_calculator.calculate(daily_returns, positions, self.test_df)
    
    def _run_random_walk(self, n_runs: int = 50) -> Dict:
        """Random walk baseline."""
        np.random.seed(42)
        all_metrics = []
        
        for _ in range(n_runs):
            positions = np.random.randint(0, 2, size=len(self.test_df))
            positions = self.position_normalizer.normalize(positions, self.test_df)
            daily_returns = self.return_calculator.calculate(positions, self.test_df)
            metrics = self.metrics_calculator.calculate(daily_returns, positions, self.test_df)
            all_metrics.append(metrics)
        
        # Find median-representative run
        returns = [m["total_return"] for m in all_metrics]
        median_return = np.median(returns)
        median_idx = np.argmin(np.abs(np.array(returns) - median_return))
        
        result = all_metrics[median_idx]
        result["_metadata"] = {
            "n_runs": n_runs,
            "median_return": float(median_return),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns))
        }
        return result
    
    def _apply_probability_weights(self, probabilities: np.ndarray) -> np.ndarray:
        """Normalize probabilities per date."""
        weights = np.zeros_like(probabilities)
        dates = self.test_df["date"].unique()
        
        for date in dates:
            mask = self.test_df["date"] == date
            date_probs = probabilities[mask]
            total_prob = np.sum(date_probs[date_probs > 0])
            
            if total_prob > 0:
                weights[mask] = np.where(
                    date_probs > 0,
                    date_probs / total_prob,
                    0
                )
        
        return weights
    
    def save_summary(self, backtest_results: Dict, output_dir: str) -> None:
        """Save backtest summary to JSON."""
        summary = {}
        for strategy_name, metrics in backtest_results.items():
            summary[strategy_name] = {
                "total_return": metrics["total_return"],
                "annualized_return": metrics["annualized_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "active_hit_rate": metrics["active_hit_rate"],
                "final_equity": metrics["final_equity"]
            }
            if "_metadata" in metrics:
                summary[strategy_name]["_metadata"] = metrics["_metadata"]
        
        with open(f"{output_dir}/backtest_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        self._print_summary(summary)
    
    def _print_summary(self, summary: Dict) -> None:
        """Print backtest summary to console."""
        print("\n" + "="*90)
        print("BACKTEST SUMMARY - STRATEGY COMPARISON")
        print("="*90)
        
        threshold_results = {k: v for k, v in summary.items() if "Threshold" in k}
        if threshold_results:
            print("\nMULTIPLE PROBABILITY THRESHOLDS:")
            print("-" * 90)
            for name in sorted(threshold_results.keys()):
                self._print_strategy_metrics(name, threshold_results[name])
        
        benchmark_results = {k: v for k, v in summary.items() if "Threshold" not in k}
        if benchmark_results:
            print("\n\nBENCHMARK STRATEGIES:")
            print("-" * 90)
            for name, metrics in benchmark_results.items():
                self._print_strategy_metrics(name, metrics)
        
        print("\n" + "="*90 + "\n")
    
    def _print_strategy_metrics(self, name: str, metrics: Dict) -> None:
        """Print metrics for a single strategy."""
        print(f"\n{name}:")
        print(f"  Total Return:     {metrics['total_return']*100:>8.2f}%")
        print(f"  Annualized Ret:   {metrics['annualized_return']*100:>8.2f}%")
        print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>8.4f}")
        print(f"  Max Drawdown:     {metrics['max_drawdown']*100:>8.2f}%")
        print(f"  Active Hit Rate:  {metrics['active_hit_rate']*100:>8.2f}%")
        print(f"  Final Equity:     ${metrics['final_equity']:>12,.2f}")
    
    def plot_results(self, backtest_results: Dict, output_dir: str) -> None:
        """Generate all plots."""
        self.plot_generator.plot_all(backtest_results, self.test_df, output_dir)
