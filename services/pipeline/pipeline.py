import pandas as pd
import time
from pathlib import Path
from typing import Tuple, Dict, Any
from services.pipeline.model_selector import ModelSelector
from services.pipeline.metrics_evaluator import MetricsEvaluator
from services.pipeline.reporter import PipelineReporter
from services.backtesting import Backtest
from services.stock import Stock
from services.transform import FeatureEngineer
from services.algorithms.base import Algorithm
from services.logger_config import get_logger


class Pipeline:
    """Main ML pipeline orchestrator.
    
    Composes: ModelSelector, MetricsEvaluator, PipelineOrchestrator, Backtest
    Workflow: Data → Features → Model Selection → Backtesting
    """
    
    def __init__(
        self,
        stock: Stock,
        algorithms: list[Algorithm],
        output_dir: str = "output",
        test_size: float = 0.20,
        wfv_train_window: int = 750,
        wfv_test_window: int = 250,
        initial_capital: float = 10000,
        transaction_cost: float = 0.0005,
        slippage: float = 0.0005,
        annual_rf_rate: float = 0.05,
        probability_thresholds: list[float] = None,
        position_sizing: str = "equal_weight",
    ):
        """Initialize pipeline with data source and algorithms."""
        self.stock = stock
        self.algorithms = algorithms
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = test_size
        self.features = FeatureEngineer()
        
        # Walk-Forward Validation parameters
        self.wfv_train_window = wfv_train_window
        self.wfv_test_window = wfv_test_window
        
        # Backtesting parameters
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.annual_rf_rate = annual_rf_rate
        self.probability_thresholds = probability_thresholds or [0.50, 0.55, 0.60, 0.65, 0.70]
        self.position_sizing = position_sizing
        
        # Initialize components
        self.model_selector = ModelSelector(wfv_train_window, wfv_test_window)
        self.metrics_evaluator = MetricsEvaluator()
        self.reporter = PipelineReporter(self.output_dir)
        self.logger = get_logger()
        
        # Timing tracking
        self.phase_times = {}
    
    def run(self, save: bool = True) -> Dict[str, Any]:
        """Execute complete pipeline: data → features → model selection → backtest.
        
        Returns:
            Dictionary with metrics and backtest results
        """
        pipeline_start = time.time()
        
        self.reporter.log_phase_start(
            "0: Data Preparation",
            "Loading historical data and engineering features"
        )
        phase_start = time.time()
        
        try:
            # Phase 0: Data preparation
            self.logger.info("Loading historical stock data...")
            raw_df = self.stock.fetch()
            self.logger.info(f"✓ Fetched {len(raw_df)} records")
            
            self.logger.info("Building features...")
            dataset, feature_cols, target_col = self.features.build(
                raw_df, self.algorithms[0].feature_profile()
            )
            self.logger.info(f"✓ Built {len(feature_cols)} features, dataset shape: {dataset.shape}")
            
            self.logger.info("Performing temporal train-test split (80-20)...")
            train_df, test_df = self._split(dataset)
            self.phase_times["0_data_preparation"] = time.time() - phase_start
            self.reporter.log_phase_end("0: Data Preparation")
            
            # Phase 1: Model selection via walk-forward validation
            self.reporter.log_phase_start(
                "1: Model Selection",
                "Using walk-forward validation with Information Coefficient"
            )
            phase_start = time.time()
            self.logger.info(f"Testing {len(self.algorithms)} algorithms...")
            best_model, best_model_name, best_score = self.model_selector.select(
                self.algorithms, train_df, feature_cols, target_col
            )
            self.phase_times["1_model_selection"] = time.time() - phase_start
            self.reporter.log_model_selection(best_model_name, best_score, {})
            self.reporter.log_phase_end("1: Model Selection")
            
            # Phase 2: Retrain best model and evaluate
            self.reporter.log_phase_start(
                "2: Evaluation",
                f"Retraining {best_model_name} on full train set"
            )
            phase_start = time.time()
            self.logger.info(f"Retraining {best_model_name} on complete train set...")
            retrain_start = time.time()
            best_model.fit(train_df, pd.DataFrame(), feature_cols, target_col)
            retrain_time = time.time() - retrain_start
            self.logger.info(f"✓ Retraining completed in {retrain_time:.2f}s")
            
            # Evaluate best model on test set (no need to re-train/evaluate others)
            self.logger.info(f"Evaluating {best_model_name} on test set...")
            results = {}
            metrics = self.metrics_evaluator.evaluate(best_model, test_df, feature_cols, target_col)
            results[best_model_name] = metrics
            self.logger.info(f"✓ Evaluation completed for best model: {best_model_name}")
            self.phase_times["2_evaluation"] = time.time() - phase_start
            self.reporter.log_phase_end("2: Evaluation")
            
            # Phase 3: Save and backtest
            if save:
                self.reporter.log_phase_start(
                    "3: Backtesting",
                    f"Final backtest on test set with {best_model_name}"
                )
                phase_start = time.time()
                self.logger.info("Saving metrics to JSON...")
                self.reporter.save_metrics(results)
                
                self.logger.info("Generating metrics comparison plot...")
                self.reporter.plot_metrics_comparison(results)
                
                self.logger.info("Running backtest with multiple strategies...")
                self._run_backtest(best_model, test_df, feature_cols)
                self.phase_times["3_backtesting"] = time.time() - phase_start
                self.reporter.log_phase_end("3: Backtesting")
            
            # Log timing summary
            total_time = time.time() - pipeline_start
            self.reporter.log_timing_summary(self.phase_times, total_time)
            
            self.logger.info("="*80)
            self.logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            self.logger.info("="*80)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            self.reporter.log_phase_end("Pipeline", "FAILED")
            raise
    
    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/test by date (temporal split)."""
        dates = df["date"].sort_values().unique()
        n = len(dates)
        train_end_idx = int(n * (1 - self.test_size))
        split_date = dates[train_end_idx]
        
        train_df = df[df["date"] < split_date].copy()
        test_df = df[df["date"] >= split_date].copy()
        
        self.logger.info(
            f"Train: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})"
        )
        self.logger.info(
            f"Test:  {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})"
        )
        
        return train_df, test_df
    
    def _run_backtest(self, best_algo: Algorithm, test_df: pd.DataFrame, 
                     features: list[str]) -> None:
        """Run backtesting with multiple probability thresholds."""
        self.logger.info("Initializing backtest engine...")
        
        # Get predictions
        y_prob = best_algo.predict_proba(test_df, features)
        
        # Run backtest
        backtest = Backtest(
            test_df,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            slippage=self.slippage,
            annual_rf_rate=self.annual_rf_rate,
            position_sizing=self.position_sizing
        )
        
        self.logger.info(f"Running strategies with probability thresholds: {self.probability_thresholds}")
        results = backtest.run_threshold_strategies(y_prob, self.probability_thresholds)
        
        # Save and plot
        self.logger.info("Saving backtest summary...")
        backtest.save_summary(results, str(self.output_dir))
        
        self.logger.info("Generating backtest visualizations...")
        backtest.plot_results(results, str(self.output_dir))
        
        self.reporter.log_backtest_results(results)

