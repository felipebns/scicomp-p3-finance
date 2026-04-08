import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import json

class Backtest:
    """Backtesting engine to compare trading strategies with realistic assumptions"""
    
    def __init__(self, test_df: pd.DataFrame, initial_capital: float = 10000,
                 transaction_cost: float = 0.0005, slippage: float = 0.0005,
                 annual_rf_rate: float = 0.05):
        self.test_df = test_df.copy()
        self.initial_capital = initial_capital
        self.tc = transaction_cost
        self.slippage = slippage
        self.annual_rf_rate = annual_rf_rate
        self.daily_rf_rate = (1 + annual_rf_rate) ** (1/252) - 1
        
    def run_all_strategies(self, model_predictions: np.ndarray) -> Dict[str, Dict]:
        """Run all benchmark strategies and model strategy"""
        results = {}
        
        results["ML Model"] = self._model_strategy(model_predictions)
        results["Buy & Hold"] = self._buy_and_hold()
        results["Fixed Income (Cash)"] = self._fixed_income()
        results["Mean Reversion"] = self._mean_reversion()
        results["Random Walk (Median)"] = self._random_walk_monte_carlo(n_runs=50)
        
        return results
        
    def _calculate_strategy_returns(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate net returns given a vector of target positions for each day.
        - positions[t]: Position taken at the CLOSE of day t.
        - Return depends on price change from t to t+1.
        - Costs are deducted when position changes.
        """
        next_returns = self.test_df["return"].shift(-1).fillna(0).values
        
        gross_returns = positions * next_returns
        
        position_changes = np.diff(positions, prepend=0)
        costs = np.abs(position_changes) * (self.tc + self.slippage)
        
        cash_interest = np.where(positions == 0, self.daily_rf_rate, 0)
        
        net_returns = gross_returns - costs + cash_interest
        return net_returns

    def _model_strategy(self, predictions: np.ndarray) -> Dict:
        """Strategy based on ML model predictions"""
        # Assume prediction is 1 (Long) or 0 (Cash)
        positions = predictions
        net_returns = self._calculate_strategy_returns(positions)
        return self._calculate_metrics(net_returns, positions)
    
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
        """Robust Random Walk calculates metrics for each run and returns empirical median path."""
        np.random.seed(42)
        all_metrics = []
        all_curves = []
        all_positions = []
        
        for _ in range(n_runs):
            positions = np.random.randint(0, 2, size=len(self.test_df))
            net_returns = self._calculate_strategy_returns(positions)
            metrics = self._calculate_metrics(net_returns, positions)
            all_metrics.append(metrics)
            all_curves.append(metrics["equity_curve"])
            all_positions.append(positions)
            
        # Obter o indíce do run cujo retorno total é o mais próximo da mediana
        returns = [m["total_return"] for m in all_metrics]
        median_idx = np.argsort(returns)[len(returns)//2]
        
        representative_metrics = all_metrics[median_idx]
        
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
        
        is_active = (positions > 0)
        active_days = np.sum(is_active)
        if active_days > 0:
            active_hit_rate = np.sum(strategy_returns[is_active] > 0) / active_days
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
        """Plot backtest results comparison with Normalized Equity"""
        dates = self.test_df["date"].values
        equity_curves = {name: data["equity_curve"] for name, data in backtest_results.items()}
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        for strategy_name, equity_curve in equity_curves.items():
            # CORREÇÃO: Normalizar pelo equity INICIAL em vez de equity_curve[0] que em alguns cenarios de caixa pode ser levemente diferente
            normalized_curve = equity_curve / self.initial_capital
            ax1.plot(dates, normalized_curve, label=strategy_name, linewidth=2, alpha=0.8)
            
        ax1.set_title("Backtest: Normalized Equity Curve Comparison", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Normalized Equity (Multiple of Initial Capital)")
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Initial Capital (1.0x)')
        
        metrics_names = ["annualized_return", "sharpe_ratio", "max_drawdown", "active_hit_rate"]
        metrics_data = {strategy: [] for strategy in backtest_results.keys()}
        
        for metric_name in metrics_names:
            for strategy_name, data in backtest_results.items():
                metrics_data[strategy_name].append(data[metric_name])
                
        metrics_df = pd.DataFrame(metrics_data, index=metrics_names).T
        
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
        
        summary = {}
        for strategy_name, metrics in backtest_results.items():
            summary[strategy_name] = {
                "total_return": float(metrics["total_return"]),
                "annualized_return": float(metrics["annualized_return"]),
                "sharpe_ratio": float(metrics["sharpe_ratio"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "active_hit_rate": float(metrics["active_hit_rate"]),
                "final_equity": float(metrics["final_equity"])
            }
            
        with open(f"{output_dir}/backtest_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("\n" + "="*70)
        print("BACKTEST SUMMARY - STRATEGY COMPARISON")
        print("="*70)
        for strategy_name, metrics in summary.items():
            print(f"\n{strategy_name}:")
            print(f"  Total Ret:     {metrics['total_return']*100:.2f}%")
            print(f"  Ann. Ret:      {metrics['annualized_return']*100:.2f}%")
            print(f"  Sharpe:        {metrics['sharpe_ratio']:.4f}")
            print(f"  Max DD:        {metrics['max_drawdown']*100:.2f}%")
            print(f"  Act. Hit Rate: {metrics['active_hit_rate']*100:.2f}%")
            print(f"  Final Equity:  ${metrics['final_equity']:,.2f}")
        print("="*70 + "\n")
