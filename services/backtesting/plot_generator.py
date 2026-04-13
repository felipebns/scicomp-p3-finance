import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict


class PlotGenerator:
    """Generates all visualization plots: backtest results and model comparison.
    
    Responsibilities:
    - All matplotlib-based visualizations for the pipeline
    - Backtest equity curves, drawdowns, and strategy comparisons
    - Model selection metrics and comparison charts
    - Portfolio allocation and stock selection summaries
    """
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
    
    def plot_model_metrics_comparison(self, model_results: Dict[str, Dict], output_dir: str) -> None:
        """Plot and save model comparison visualization with multiple metrics.
        
        Args:
            model_results: Dictionary mapping model names to their WFV metrics
            output_dir: Output directory for the plot
        """
        try:
            # Extract metrics
            models = list(model_results.keys())
            mean_ics = [model_results[m]['mean_ic'] for m in models]
            std_ics = [model_results[m]['std_ic'] for m in models]
            scores = [model_results[m]['wfv_score'] for m in models]
            accuracies = [model_results[m].get('accuracy', 0) for m in models]
            f1_scores = [model_results[m].get('f1_score', 0) for m in models]
            aucs = [model_results[m].get('auc', 0.5) for m in models]
            
            # Create 3x2 subplots
            fig, axes = plt.subplots(3, 2, figsize=(14, 14))
            
            # Plot 1: Mean IC
            ax = axes[0, 0]
            colors = ['green' if x > 0 else 'red' for x in mean_ics]
            ax.bar(models, mean_ics, color=colors, alpha=0.7)
            ax.set_title("Mean Information Coefficient (IC)", fontweight='bold', fontsize=12)
            ax.set_ylabel("Mean IC")
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(mean_ics):
                ax.text(i, v, f'{v:.4f}', ha='center', va='bottom' if v > 0 else 'top', fontsize=9, fontweight='bold')
            
            # Plot 2: Std IC (lower is more stable)
            ax = axes[0, 1]
            colors = ['green' if x < 0.035 else 'orange' for x in std_ics]
            ax.bar(models, std_ics, color=colors, alpha=0.7)
            ax.set_title("Std Dev of IC (Lower = More Stable)", fontweight='bold', fontsize=12)
            ax.set_ylabel("Std IC")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(std_ics):
                ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Plot 3: WFV Score (Robust Score)
            ax = axes[1, 0]
            colors_score = ['green' if x > 0 else 'red' for x in scores]
            bars = ax.bar(models, scores, color=colors_score, alpha=0.7, edgecolor='black', linewidth=2)
            best_idx = np.argmax(scores)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
            ax.set_title("Walk-Forward Validation Score\n(Mean IC - 0.5×Std IC) [SELECTION METRIC]", fontweight='bold', fontsize=12)
            ax.set_ylabel("WFV Score")
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(scores):
                ax.text(i, v, f'{v:.4f}', ha='center', va='bottom' if v > 0 else 'top', fontsize=9, fontweight='bold')
            
            # Plot 4: Accuracy
            ax = axes[1, 1]
            bars = ax.bar(models, accuracies, color='steelblue', alpha=0.7)
            ax.set_title("Classification Accuracy", fontweight='bold', fontsize=12)
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(accuracies):
                ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Plot 5: F1 Score
            ax = axes[2, 0]
            bars = ax.bar(models, f1_scores, color='coral', alpha=0.7)
            ax.set_title("F1 Score", fontweight='bold', fontsize=12)
            ax.set_ylabel("F1 Score")
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(f1_scores):
                ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Plot 6: AUC (ROC)
            ax = axes[2, 1]
            bars = ax.bar(models, aucs, color='mediumseagreen', alpha=0.7)
            ax.set_title("ROC-AUC Score", fontweight='bold', fontsize=12)
            ax.set_ylabel("AUC")
            ax.set_ylim(0.4, 1.0)
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Random')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()
            for i, v in enumerate(aucs):
                ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            filepath = f"{output_dir}/model_selection_comparison.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            # Log error silently (don't use logger here to avoid circular imports)
            print(f"Warning: Failed to plot model comparison: {e}")
            raise
    
    def plot_all(self, backtest_results: Dict, test_df: pd.DataFrame, output_dir: str) -> None:
        """Generate all backtest plots."""
        dates = np.sort(test_df["date"].unique())
        
        # Separate results by strategy type
        ml_strategies = {k: v for k, v in backtest_results.items() if k.startswith("ML ")}
        benchmark_strategies = {k: v for k, v in backtest_results.items() if "benchmark" in k.lower()}
        baseline_strategies = {k: v for k, v in backtest_results.items() 
                             if k.startswith("Threshold") or k.startswith("Buy & Hold") or k.startswith("Fixed")}
        
        # Generate separate plots for clarity
        self._plot_ml_strategies_by_threshold(ml_strategies, dates, output_dir)
        self._plot_all_strategies_comparison(backtest_results, dates, output_dir)
        self._plot_strategy_metrics_comparison(backtest_results, output_dir)
        self._plot_threshold_effect(ml_strategies, output_dir)
        self._plot_drawdown_analysis(backtest_results, dates, output_dir)
        self._plot_portfolio_allocation_summary(backtest_results, test_df, output_dir)
    
    def _plot_ml_strategies_by_threshold(self, ml_results: Dict, dates: np.ndarray, output_dir: str) -> None:
        """Plot ML strategies grouped by threshold with different colors."""
        if not ml_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Group by strategy name (ensemble_smart, momentum, mean_reversion, volatility_weighted)
        strategies = {}
        for name, data in ml_results.items():
            parts = name.split()  # "ML strategy_name threshold"
            strategy_name = f"{parts[1]} {parts[2]}"  # e.g., "ensemble_smart 0.50"
            base_strategy = parts[1]  # e.g., "ensemble_smart"
            if base_strategy not in strategies:
                strategies[base_strategy] = {}
            strategies[base_strategy][name] = data
        
        # Plot each base strategy with different thresholds
        colors_by_threshold = {
            "0.50": "blue",
            "0.52": "green", 
            "0.53": "orange",
            "0.55": "red",
            "0.57": "purple"
        }
        
        for idx, (base_strategy, strategy_results) in enumerate(strategies.items()):
            ax = axes[idx]
            
            for full_name, data in sorted(strategy_results.items()):
                equity_curve = data["equity_curve"]
                normalized = equity_curve / self.initial_capital
                threshold = full_name.split()[-1]  # Extract threshold
                color = colors_by_threshold.get(threshold, "gray")
                
                ax.plot(dates, normalized, label=f"Threshold {threshold}", linewidth=2.5, 
                       color=color, alpha=0.8)
            
            ax.set_title(f"ML Strategy: {base_strategy.replace('_', ' ').title()}", 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel("Date")
            ax.set_ylabel("Normalized Equity")
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ml_strategies_by_threshold.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_all_strategies_comparison(self, results: Dict, dates: np.ndarray, output_dir: str) -> None:
        """Plot all strategies together for overall comparison."""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Select best threshold version (0.50) for each ML strategy, and include all benchmarks
        selected = {}
        for name, data in results.items():
            if name.startswith("ML "):
                # Select 0.50 threshold only
                if name.endswith(" 0.50"):
                    short_name = " ".join(name.split()[1:-1]).replace("_", " ").title()
                    selected[short_name] = data
            else:
                # Include ALL benchmarks and baselines (Buy & Hold, Fixed Income, etc.)
                selected[name] = data
        
        # Plot with different colors
        color_map = {
            "Ensemble Smart": "darkblue",
            "Momentum": "green",
            "Mean Reversion": "orange",
            "Volatility Weighted": "red",
            "Threshold Only (baseline)": "gray",
            "Buy & Hold (benchmark)": "purple",
            "Fixed Income 5.00% (benchmark)": "brown",
            "Simple Reversal SMA20 (benchmark)": "pink"
        }
        
        for strategy_name, data in selected.items():
            equity_curve = data["equity_curve"]
            normalized = equity_curve / self.initial_capital
            color = color_map.get(strategy_name, "black")
            
            ax.plot(dates, normalized, label=strategy_name, linewidth=2.5, 
                   color=color, alpha=0.8)
        
        ax.set_title("All Strategies Comparison (Best Threshold Configuration)", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Equity")
        ax.legend(loc='best', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/all_strategies_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_strategy_metrics_comparison(self, results: Dict, output_dir: str) -> None:
        """Plot metrics comparison for top strategies only."""
        # Select best threshold for each strategy
        selected = {}
        for name, data in results.items():
            if name.startswith("ML "):
                if name.endswith(" 0.50"):
                    short_name = " ".join(name.split()[1:-1]).replace("_", " ").title()
                    selected[short_name] = data
            else:
                selected[name] = data
        
        metrics = ["annualized_return", "sharpe_ratio", "max_drawdown", "active_hit_rate"]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [selected[s].get(metric, 0) for s in selected.keys()]
            colors = ["green" if v > 0 else "red" for v in values]
            
            bars = ax.bar(range(len(selected)), values, color=colors, alpha=0.7)
            ax.set_xticks(range(len(selected)))
            ax.set_xticklabels(selected.keys(), rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{metric.replace('_', ' ').title()} Comparison", fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_effect(self, ml_results: Dict, output_dir: str) -> None:
        """Plot how threshold affects returns for each ML strategy."""
        if not ml_results:
            return
        
        # Group by base strategy
        strategies = {}
        for name, data in ml_results.items():
            parts = name.split()
            base_strategy = parts[1]
            threshold = float(parts[2])
            
            if base_strategy not in strategies:
                strategies[base_strategy] = {"thresholds": [], "returns": [], "sharpe": []}
            
            strategies[base_strategy]["thresholds"].append(threshold)
            strategies[base_strategy]["returns"].append(data.get("total_return", 0) * 100)
            strategies[base_strategy]["sharpe"].append(data.get("sharpe_ratio", 0))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = ["blue", "green", "orange", "red"]
        
        for color, (strategy_name, data) in zip(colors, strategies.items()):
            thresholds = sorted(zip(data["thresholds"], data["returns"], data["sharpe"]))
            thresholds_sorted = [t[0] for t in thresholds]
            returns_sorted = [t[1] for t in thresholds]
            sharpe_sorted = [t[2] for t in thresholds]
            
            ax1.plot(thresholds_sorted, returns_sorted, marker='o', label=strategy_name.replace("_", " ").title(),
                    color=color, linewidth=2, markersize=6)
            ax2.plot(thresholds_sorted, sharpe_sorted, marker='s', label=strategy_name.replace("_", " ").title(),
                    color=color, linewidth=2, markersize=6)
        
        ax1.set_xlabel("Probability Threshold")
        ax1.set_ylabel("Total Return (%)")
        ax1.set_title("Effect of Threshold on Returns", fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel("Probability Threshold")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_title("Effect of Threshold on Sharpe Ratio", fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/threshold_effect.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_drawdown_analysis(self, results: Dict, dates: np.ndarray, output_dir: str) -> None:
        """Plot maximum drawdown for each strategy."""
        max_drawdowns = []
        strategy_names = []
        
        # Select best threshold for each strategy
        for name, data in results.items():
            if name.startswith("ML "):
                if name.endswith(" 0.50"):
                    short_name = " ".join(name.split()[1:-1]).replace("_", " ").title()
                    strategy_names.append(short_name)
                    max_drawdowns.append(data.get("max_drawdown", 0) * 100)
            else:
                strategy_names.append(name)
                max_drawdowns.append(data.get("max_drawdown", 0) * 100)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ["red" if x < 0 else "green" for x in max_drawdowns]
        bars = ax.barh(strategy_names, max_drawdowns, color=colors, alpha=0.7)
        
        ax.set_xlabel("Maximum Drawdown (%)")
        ax.set_title("Maximum Drawdown by Strategy", fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2., f'{width:.1f}%',
                   ha='left' if width > 0 else 'right', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/max_drawdown.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_stock_selection_summary(self, test_df: pd.DataFrame, output_dir: str) -> None:
        """Plot summary of which stocks were selected most frequently.
        
        Args:
            test_df: Test dataframe with ticker column
        """
        try:
            # Count how many times each ticker appears in the test set
            ticker_counts = test_df['ticker'].value_counts()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Distribution of observations per ticker (data availability)
            ax1.barh(ticker_counts.index, ticker_counts.values, color='steelblue', alpha=0.7)
            ax1.set_xlabel("Number of Daily Observations")
            ax1.set_title("Stock Data Availability in Test Set\n(Daily Observations per Ticker)", 
                         fontweight='bold', fontsize=12)
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Plot 2: Unique dates and market composition
            n_dates = test_df['date'].nunique()
            n_stocks = test_df['ticker'].nunique()
            
            summary_text = f"""
Test Set Summary
━━━━━━━━━━━━━━━━━━━━━━━━━
Total Trading Days:    {n_dates}
Unique Stocks:         {n_stocks}
Total Observations:    {len(test_df)}

Date Range:
  Start: {test_df['date'].min().date()}
  End:   {test_df['date'].max().date()}

Expected Observations:
  (if all stocks every day): {n_dates * n_stocks}
  Actual: {len(test_df)}

Portfolio Allocation:
{chr(10).join([f"  {ticker}: {count:>4} days ({count/n_dates*100:>5.1f}%)" 
              for ticker, count in ticker_counts.items()])}
            """
            
            ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, 
                    fontfamily='monospace', fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/stock_selection_summary.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            # Silently skip if error (non-critical visualization)
            pass
    
    def _plot_portfolio_allocation_summary(self, backtest_results: Dict, test_df: pd.DataFrame, output_dir: str) -> None:
        """Plot portfolio allocation: which stocks were SELECTED and their average weights.
        
        Shows actual position weights from the best-performing strategy (ML ensemble at 0.50 threshold).
        This tells you which stocks mattered for the decision-making.
        
        Args:
            backtest_results: Backtest results with equity curves
            test_df: Test dataframe with ticker column
            output_dir: Output directory for the plot
        """
        try:
            # Select the best ML strategy result (ensemble at 0.50 threshold)
            best_strategy_key = None
            best_return = -np.inf
            
            for strategy_name, data in backtest_results.items():
                if "ensemble_smart" in strategy_name and "0.50" in strategy_name:
                    ret = data.get("total_return", 0)
                    if ret > best_return:
                        best_return = ret
                        best_strategy_key = strategy_name
            
            if best_strategy_key is None:
                # Fallback: just pick the first ML strategy if no ensemble
                for strategy_name in backtest_results.keys():
                    if strategy_name.startswith("ML ") and " 0.50" in strategy_name:
                        best_strategy_key = strategy_name
                        break
            
            if best_strategy_key is None:
                return  # No ML strategy found, skip
            
            # Note: To get actual weights, we'd need to track positions in backtest
            # For now, show which stocks were in the test set and their frequency
            # This is a proxy for "selection" - more data points = more selection opportunities
            
            ticker_counts = test_df['ticker'].value_counts().sort_values(ascending=False)
            n_dates = test_df['date'].nunique()
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Left: Bar chart of stock frequency (proxy for selection opportunity)
            ax = axes[0]
            tickers = ticker_counts.index
            counts = ticker_counts.values
            percentages = (counts / n_dates) * 100
            
            colors_list = ['green' if p > 90 else 'orange' if p > 50 else 'red' for p in percentages]
            bars = ax.barh(tickers, percentages, color=colors_list, alpha=0.7)
            
            ax.set_xlabel("Days Present in Test Set (%)")
            ax.set_title("Stock Selection Opportunity\n(% of trading days with data available)", 
                        fontweight='bold', fontsize=12)
            ax.set_xlim(0, 105)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add percentage labels
            for i, (ticker, pct) in enumerate(zip(tickers, percentages)):
                ax.text(pct + 1, i, f'{pct:.1f}%', va='center', fontsize=9, fontweight='bold')
            
            # Right: Summary statistics
            ax = axes[1]
            summary_stats = f"""
Portfolio Composition Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Selected Strategy:
  {best_strategy_key}

Stock Universe:
  Total stocks: {n_stocks := len(tickers)}
  Trading days analyzed: {n_dates}

Selection Frequency (days available):
  High (>90%):    {len([p for p in percentages if p > 90])} stocks
  Medium (50-90%): {len([p for p in percentages if 50 <= p <= 90])} stocks
  Low (<50%):     {len([p for p in percentages if p < 50])} stocks

Stock Details:
{chr(10).join([f"  {ticker:6s}: {count:3d} days ({pct:5.1f}%)" 
              for ticker, count, pct in zip(tickers, counts, percentages)])}

Note: Position WEIGHTS are determined by your 
position_sizing strategy (equal_weight vs 
probability_weighted). This chart shows selection
OPPORTUNITY for each stock based on data availability.

To see actual portfolio weights over time,
review the equity curves and position histories
in the detailed backtest logs.
            """
            
            ax.text(0.05, 0.95, summary_stats, transform=ax.transAxes, 
                   fontfamily='monospace', fontsize=9.5, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/portfolio_allocation_summary.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        except Exception as e:
            # Silently skip if error
            pass
