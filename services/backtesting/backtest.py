import json
import pandas as pd
import numpy as np
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from services.backtesting.metrics_calculator import MetricsCalculator
from services.backtesting.position_normalizer import PositionNormalizer
from services.backtesting.return_calculator import ReturnCalculator
from services.backtesting.plot_generator import PlotGenerator
from services.strategies import (
    MomentumStrategy, MeanReversionStrategy, 
    VolatilityWeightedStrategy, EnsembleSmartStrategy,
    ThresholdStrategy, BuyAndHoldStrategy, FixedIncomeStrategy,
    SimpleReversalStrategy
)


class Backtest:
    """Backtesting engine for multi-asset trading strategies."""
    
    def __init__(self, test_df: pd.DataFrame, initial_capital: float = 10000,
                 transaction_cost: float = 0.0005, slippage: float = 0.0005,
                 annual_rf_rate: float = 0.05, position_sizing: str = "equal_weight",
                 position_selection: str = "top_5",
                 allocation_mode: str = "full_deployment",
                 purchase_threshold: float = 0.50,
                 threshold_workers: int = 3):
        self.test_df = test_df.copy()
        self.initial_capital = initial_capital
        self.annual_rf_rate = annual_rf_rate
        self.position_sizing = position_sizing
        self.position_selection = position_selection
        self.allocation_mode = allocation_mode
        self.purchase_threshold = purchase_threshold
        self.threshold_workers = threshold_workers
        
        # Initialize components
        self.position_normalizer = PositionNormalizer()
        self.return_calculator = ReturnCalculator(transaction_cost, slippage, annual_rf_rate)
        self.metrics_calculator = MetricsCalculator(initial_capital, annual_rf_rate)
        self.plot_generator = PlotGenerator(initial_capital)
        
        # Initialize strategies
        self.strategies = {
            # Smart strategies (ML + technical)
            "momentum": MomentumStrategy(),
            "mean_reversion": MeanReversionStrategy(),
            "volatility_weighted": VolatilityWeightedStrategy(),
            "ensemble_smart": EnsembleSmartStrategy(),
            # Simple/baseline strategies
            "threshold": ThresholdStrategy(),
            "buy_and_hold": BuyAndHoldStrategy(),
            "fixed_income": FixedIncomeStrategy(),
            "simple_reversal": SimpleReversalStrategy()
        }
    
    def run_threshold_strategies(self, model_probabilities: np.ndarray, 
                                 thresholds: list[float] = None) -> Dict[str, Dict]:
        """Run all 8 strategies with multiple probability thresholds in parallel."""
        if thresholds is None:
            thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
        
        results = {}
        
        # All strategies to test (4 intelligent + 4 benchmarks)
        ml_strategies = ["ensemble_smart", "momentum", "mean_reversion", "volatility_weighted"]
        
        # Test all ML strategies with all thresholds in parallel
        with ThreadPoolExecutor(max_workers=self.threshold_workers) as executor:
            futures = {}
            
            # Submit all ML strategy + threshold combinations
            for strategy_name in ml_strategies:
                for threshold in thresholds:
                    future = executor.submit(
                        self._run_strategy, model_probabilities, threshold, strategy_name
                    )
                    futures[future] = (strategy_name, threshold)
            
            # Collect results
            for future in futures:
                strategy_name, threshold = futures[future]
                results[f"ML {strategy_name} {threshold:.2f}"] = future.result()
        
        # Add baseline strategies (sequential, don't need parallelization)
        results["Threshold Only (baseline)"] = \
            self._run_strategy(model_probabilities, 0.5, "threshold")
        results["Buy & Hold (benchmark)"] = \
            self._run_strategy(model_probabilities, 0.5, "buy_and_hold")
        results[f"Fixed Income {self.annual_rf_rate:.2%} (benchmark)"] = \
            self._run_strategy(model_probabilities, 0.5, "fixed_income")
        results["Simple Reversal SMA20 (benchmark)"] = \
            self._run_strategy(model_probabilities, 0.5, "simple_reversal")
        
        return results
    
    def _run_strategy(self, probabilities: np.ndarray, threshold: float, 
                      strategy_name: str) -> Dict:
        """Run a specific strategy by name."""
        strategy = self.strategies.get(strategy_name)
        
        if strategy is None:
            # Fallback to threshold strategy
            positions = (probabilities > threshold).astype(int)
        else:
            # Get per-ticker data for strategy application
            positions = np.zeros(len(self.test_df))
            tickers = self.test_df["ticker"].unique()
            
            for ticker in tickers:
                mask = self.test_df["ticker"] == ticker
                ticker_df = self.test_df[mask].reset_index(drop=True)
                ticker_probs = probabilities[mask]
                
                # Apply strategy
                ticker_positions = strategy.apply(ticker_df, ticker_probs, threshold)
                positions[mask] = ticker_positions
        
        # Apply position selection filter (e.g., top_5)
        # Skip for benchmarks that should hold everything or nothing
        """
        Note: Buy and hold should not use weights from the model, keep as neutral baseline
        Meaning it should hold all stocks equally regardless of probabilities, to show pure buy-and-hold performance.
        """
        if strategy_name not in ["fixed_income", "buy_and_hold"]:
            positions = self._apply_position_selection(positions, probabilities)
        
        if self.position_sizing == "probability_weighted" and strategy_name not in ["fixed_income", "buy_and_hold", "simple_reversal"]:
            # Weight the binary positions by their probabilities
            # This respects the strategy while weighting by confidence
            positions = positions * probabilities  # Element-wise multiply
            positions = self._apply_probability_weights(positions)
        
        positions = self.position_normalizer.normalize(
            positions, self.test_df, 
            allocation_mode=self.allocation_mode,
            purchase_threshold=self.purchase_threshold
        )
        daily_returns = self.return_calculator.calculate(positions, self.test_df)
        metrics = self.metrics_calculator.calculate(daily_returns, positions, self.test_df)
        
        # Store position information by ticker for visualization
        metrics["positions_by_ticker"] = self._aggregate_positions_by_ticker(positions)
        
        return metrics
    
    def _aggregate_positions_by_ticker(self, positions: np.ndarray) -> Dict[str, Dict]:
        """Aggregate positions by ticker to show final weight and historical metrics.
        
        Returns:
            Dict mapping ticker → {final_weight, avg_weight, selection_days, selection_rate}
        """
        positions_df = self.test_df.copy()
        positions_df["_position"] = positions
        
        result = {}
        last_date = positions_df["date"].max()
        
        for ticker in positions_df["ticker"].unique():
            ticker_data = positions_df[positions_df["ticker"] == ticker]
            total_days = len(ticker_data)
            active_days = np.sum(ticker_data["_position"] > 0)
            avg_weight = ticker_data["_position"].mean()
            
            # Get final weight (last day of backtest)
            last_day_data = ticker_data[ticker_data["date"] == last_date]
            final_weight = float(last_day_data["_position"].values[0]) if len(last_day_data) > 0 else 0.0
            
            result[ticker] = {
                "final_weight": final_weight,  # Weight to use in production
                "avg_position_weight": float(avg_weight),  # Historical average
                "days_selected": int(active_days),
                "total_days": int(total_days),
                "selection_rate": float(active_days / total_days) if total_days > 0 else 0.0
            }
        
        return result
    
    def _apply_position_selection(self, positions: np.ndarray, 
                                  probabilities: np.ndarray) -> np.ndarray:
        """Filter positions based on position_selection strategy (e.g., top_5).
        
        Args:
            positions: Position array (0 or 1 for each row)
            probabilities: Model probability for each row
            
        Returns:
            Filtered position array with TOP-N selection applied
        """
        if self.position_selection == "all":
            return positions
        
        # Parse position selection method (e.g., "top_5" → 5)
        if not self.position_selection.startswith("top_"):
            return positions
        
        try:
            n_top = int(self.position_selection.split("_")[1])
        except (IndexError, ValueError):
            return positions
        
        filtered_positions = positions.copy()
        dates = self.test_df["date"].unique()
        
        for date in dates:
            mask = self.test_df["date"] == date
            date_positions = positions[mask.values]  # Convert mask to numpy array
            date_probs = probabilities[mask.values]
            
            # Only apply filter if there are positions to take
            n_positions = np.sum(date_positions > 0)
            
            if n_positions > n_top:
                # Get indices of top N positions by probability (local to this date)
                position_indices = np.where(date_positions > 0)[0]  # Local indices within date
                position_probs = date_probs[position_indices]
                
                # Sort by probability descending, take top N (still local indices)
                local_top_indices = position_indices[np.argsort(-position_probs)[:n_top]]
                
                # Zero out all positions for this date
                filtered_positions[mask.values] = 0
                
                # Re-add only the top N (using local indices)
                global_indices = np.where(mask.values)[0][local_top_indices]
                filtered_positions[global_indices] = 1
        
        return filtered_positions
    
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
            if "positions_by_ticker" in metrics:
                summary[strategy_name]["positions_by_ticker"] = metrics["positions_by_ticker"]
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
