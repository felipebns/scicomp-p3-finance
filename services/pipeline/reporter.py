import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict
from services.log.logger_config import get_logger


class PipelineReporter:
    """Handles pipeline logging and result saving (NOT plotting).
    
    Responsibilities:
    - Save evaluation metrics to JSON files
    - Log pipeline events and results to logger
    - Manage output directory structure
    
    Note: All plotting (model metrics, backtest results) is handled by PlotGenerator.
    This is purely for logging and non-visualization result archival.
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

