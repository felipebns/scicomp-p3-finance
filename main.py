from multiprocessing import cpu_count
from datetime import datetime, timezone
from services.stock.stock import Stock
from services.pipeline import Pipeline
from services.algorithms.svc import SVCAlgorithm
from services.algorithms.random_forest import RandomForestAlgorithm
from services.algorithms.ensemble import EnsembleClassificationAlgorithm
from services.algorithms.logistic_regression import LogisticRegressionAlgorithm
from services.log.logger_config import setup_logging, get_logger

from config.config import CONFIG

if __name__ == "__main__":
    # Setup logging system
    logger = setup_logging(log_dir="logs")
    logger.info("="*80)
    logger.info("STARTING CLASSIFICATION PIPELINE WITH ENSEMBLE LEARNING")
    logger.info("="*80)
    
    end_date = datetime.now(timezone.utc).date().isoformat()
    
    logger.info(f"Fetching historical data for {len(CONFIG['tickers'])} tickers since {CONFIG['start_date']}...")
    stock = Stock(tickers=CONFIG["tickers"], start=CONFIG["start_date"], end=end_date)
    logger.info(f"✓ Stock data initialized")

    logger.info(f"Initializing {4} ML algorithms...")
    models = [
        LogisticRegressionAlgorithm(**CONFIG["model_params"]["LogisticRegression"]),
        SVCAlgorithm(**CONFIG["model_params"]["SVC"]),
        RandomForestAlgorithm(**CONFIG["model_params"]["RandomForest"]),
        EnsembleClassificationAlgorithm()
    ]
    logger.info(f"✓ Algorithms initialized: {[m.name() for m in models]}")

    logger.info("\nConfiguration:")
    logger.info(f"  Tickers: {CONFIG['tickers']} ({len(CONFIG['tickers'])} stocks)")
    logger.info(f"  Position Sizing: {CONFIG['position_sizing']}")
    logger.info(f"  Position Selection: {CONFIG['position_selection']} (top N stocks per day)")
    logger.info(f"  Allocation Mode: {CONFIG['allocation_mode']}")
    logger.info(f"  Purchase Threshold: {CONFIG['purchase_threshold']:.2%}")
    logger.info(f"  Transaction Cost: {CONFIG['transaction_cost']:.4%}")
    logger.info(f"  Initial Capital: ${CONFIG['initial_capital']}")
    logger.info(f"  WFV Windows: train={CONFIG['wfv_train_window']}, test={CONFIG['wfv_test_window']}")
    logger.info(f"  Probability Thresholds: {CONFIG['probability_thresholds']}")
    
    # Calculate parallelization settings automatically based on CPU count and task count
    n_algorithms = 4
    n_strategies = 8
    n_thresholds = len(CONFIG['probability_thresholds'])
    n_cpu = cpu_count()
    
    # Auto-calculate optimal workers
    parallelization = {
        "algorithm_selection": min(n_algorithms, max(1, n_cpu // 2)),  # Use half CPUs for algorithms
        "fold_evaluation": min(14, max(2, n_cpu - 1)),                  # Use N-1 CPUs for folds (leave 1 free)
        "threshold_testing": min(n_strategies * n_thresholds, n_cpu),  # Use up to all CPUs for threshold testing
    }
    CONFIG["parallelization"] = parallelization
    
    logger.info(f"\n  Backtesting will test ALL {n_strategies} STRATEGIES with {n_thresholds} thresholds each")
    logger.info(f"  Total combinations: {n_strategies} strategies × {n_thresholds} thresholds = {n_strategies * n_thresholds} tests")
    logger.info(f"\n  Parallelization (CPU cores available: {n_cpu}):")
    logger.info(f"    - Algorithm selection: {parallelization['algorithm_selection']} workers (for {n_algorithms} algorithms)")
    logger.info(f"    - Fold evaluation: {parallelization['fold_evaluation']} workers (for 14 WFV folds)")
    logger.info(f"    - Threshold testing: {parallelization['threshold_testing']} workers (for {n_strategies * n_thresholds} combinations)")

    pipeline = Pipeline(
        stock=stock, 
        algorithms=models, 
        output_dir=CONFIG["output_dir"],
        test_size=CONFIG["test_size"],
        wfv_train_window=CONFIG["wfv_train_window"],
        wfv_test_window=CONFIG["wfv_test_window"],
        initial_capital=CONFIG["initial_capital"],
        transaction_cost=CONFIG["transaction_cost"],
        slippage=CONFIG["slippage"],
        annual_rf_rate=CONFIG["annual_rf_rate"],
        probability_thresholds=CONFIG["probability_thresholds"],
        position_sizing=CONFIG["position_sizing"],
        position_selection=CONFIG["position_selection"],
        allocation_mode=CONFIG["allocation_mode"],
        purchase_threshold=CONFIG["purchase_threshold"],
        parallelization=CONFIG["parallelization"],
    )

    try:
        results = pipeline.run(save=True)
        logger.info("All results saved to 'output/' and 'logs/' directories")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise

"""TODOs"""

"""Centralize logging reporter in new folder, all responsabilities should be there"""
"""Need to create test files, too many things can break now"""
"""
Check backtesting, i might be multiplying the same starting number by the return, is this wrong ?
Should i always multiply the most recent result by the newest return ? how is it right now ?
Right now it looks like each day is independent from each other
"""
"""Understand better my code"""
"""Undestand fully the strategies, create new ones"""
"""Implement ways to run only parts of the code"""
"""Why does mean reversion need threshold ? can it just not have ? adjust periods..."""
"""Put clearer parameter usage of strategies in README"""

"""Good to have"""

"""Test what is more effective, full deployment or cash fallbacks"""
"""Stocks selection, more than 5 causes overfitting, need to think of a way to get "similar" stocks to diversify"""
"""Test different algorithms (possible deep learning ? LSTM, CNN, GRU, XGBoost, etc.)"""
"""Explore more probabilities to chose best ML model, using strategy, etc. (not only IC)"""
"""Test more buy/sell strategies"""
"""Validation fine tunning, find right number of windows days..."""
"""Parameter tuning/more features ?"""
"""Encapsulating framework"""
"""Possible signal decomposotion"""