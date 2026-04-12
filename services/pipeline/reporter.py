import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from typing import Dict
from services.logger_config import get_logger


class PipelineReporter:
    """Handles pipeline output operations: saving, plotting, and reporting results.
    
    Responsibilities:
    - Save evaluation metrics to JSON files
    - Generate and save visualization plots
    - Log pipeline events and results to logger
    - Manage output directory structure
    
    Note: This is an OUTPUT HANDLER, not an orchestrator. It doesn't coordinate
    components or manage execution flow - it only handles result reporting.
    """
    
    def __init__(self, output_dir: Path):
        """Initialize reporter with output directory.
        
        Args:
            output_dir: Path to output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
        self.logger.info(f"PipelineReporter initialized with output_dir: {self.output_dir}")
    
    def save_metrics(self, metrics: Dict) -> None:
        """Save classification metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics from best model evaluation
            Note: Only saves best model metrics (performance optimization)
        """
        try:
            filepath = self.output_dir / "classification_metrics.json"
            filepath.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            self.logger.info(f"✓ Metrics saved to {filepath}")
            self.logger.debug(f"Metrics content: {metrics}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}", exc_info=True)
            raise
    
    def plot_metrics_comparison(self, results: Dict[str, Dict]) -> None:
        """Plot and save model metrics comparison visualization.
        
        Args:
            results: Dictionary mapping model names to their metrics
        """
        try:
            self.logger.info(f"Generating metrics comparison plot for {len(results)} models")
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
            
            filepath = self.output_dir / "metrics_comparison.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.info(f"✓ Metrics comparison plot saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to plot metrics comparison: {e}", exc_info=True)
            raise
    
    def log_phase_start(self, phase_name: str, description: str = "") -> None:
        """Log the start of a pipeline phase.
        
        Args:
            phase_name: Name of the phase
            description: Optional phase description
        """
        message = f"[Phase: {phase_name}] Starting"
        if description:
            message += f" - {description}"
        self.logger.info("=" * 80)
        self.logger.info(message)
        self.logger.info("=" * 80)
    
    def log_phase_end(self, phase_name: str, status: str = "SUCCESS") -> None:
        """Log the end of a pipeline phase.
        
        Args:
            phase_name: Name of the phase
            status: Status of completion (SUCCESS, FAILED, etc.)
        """
        self.logger.info(f"[Phase: {phase_name}] Completed - Status: {status}")
    
    def log_model_selection(self, model_name: str, score: float, metrics: Dict) -> None:
        """Log model selection results.
        
        Args:
            model_name: Name of selected model
            score: Model score
            metrics: Model metrics dictionary
        """
        self.logger.info(f"✓ Best model selected: {model_name} (Score: {score:.6f})")
        self.logger.debug(f"Model metrics: {metrics}")
    
    def log_backtest_results(self, strategy_results: Dict) -> None:
        """Log backtest results for all strategies.
        
        Args:
            strategy_results: Dictionary of backtest results per strategy
        """
        self.logger.info(f"Backtest completed for {len(strategy_results)} strategies")
        for strategy_name, results in strategy_results.items():
            self.logger.info(
                f"  {strategy_name}: Return={results.get('total_return', 0):.2%}, "
                f"Sharpe={results.get('sharpe_ratio', 0):.4f}, "
                f"MaxDD={results.get('max_drawdown', 0):.2%}"
            )
    
    def log_timing_summary(self, phase_times: Dict[str, float], total_time: float) -> None:
        """Log a summary of execution times for all phases.
        
        Args:
            phase_times: Dictionary mapping phase names to their durations
            total_time: Total pipeline execution time
        """
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("EXECUTION TIME SUMMARY")
        self.logger.info("="*80)
        
        for phase, duration in phase_times.items():
            phase_name = phase.replace("_", " ").title()
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            self.logger.info(f"  {phase_name:.<50} {duration:>7.2f}s ({percentage:>5.1f}%)")
        
        self.logger.info("-"*80)
        self.logger.info(f"  {'Total Execution Time':.<50} {total_time:>7.2f}s (100.0%)")
        self.logger.info("="*80)

