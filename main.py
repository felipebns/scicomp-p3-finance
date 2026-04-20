from multiprocessing import cpu_count
from datetime import datetime, timezone
from services.stock.stock import Stock
from services.pipeline import Pipeline
from services.algorithms.svc import SVCAlgorithm
from services.algorithms.random_forest import RandomForestAlgorithm
from services.algorithms.ensemble import EnsembleClassificationAlgorithm
from services.algorithms.logistic_regression import LogisticRegressionAlgorithm
from services.log.logger_config import setup_logging, get_logger
from services.log.reporters import ApplicationReporter

from config.config import CONFIG

if __name__ == "__main__":
    # Setup logging system
    logger = setup_logging(log_dir="logs")
    app_reporter = ApplicationReporter(output_dir=CONFIG["output_dir"])
    
    # Log startup
    app_reporter.log_startup()
    
    end_date = datetime.now(timezone.utc).date().isoformat()
    
    # Log data initialization
    app_reporter.log_data_initialization(len(CONFIG['tickers']), CONFIG['start_date'])
    stock = Stock(tickers=CONFIG["tickers"], start=CONFIG["start_date"], end=end_date)

    # Log algorithms initialization
    model_names = ['Logistic Regression', 'Support Vector Classifier', 'Random Forest', 'Hybrid Ensemble']
    app_reporter.log_algorithms_initialization(model_names)
    models = [
        LogisticRegressionAlgorithm(**CONFIG["model_params"]["LogisticRegression"]),
        SVCAlgorithm(**CONFIG["model_params"]["SVC"]),
        RandomForestAlgorithm(**CONFIG["model_params"]["RandomForest"]),
        EnsembleClassificationAlgorithm()
    ]

    # Log configuration
    app_reporter.log_configuration(CONFIG)
    
    # Parallelization settings for SLURM cluster
    # NOTE: 
    # - RandomForest uses all CPUs via n_jobs=-1 (via scikit-learn joblib)
    # - ProcessPoolExecutor parallelism disabled (overhead too high for small tasks)
    # - NumPy/MKL will use OMP_NUM_THREADS from SLURM environment
    n_algorithms = 4
    n_strategies = 8
    n_thresholds = len(CONFIG['probability_thresholds'])
    n_cpu = cpu_count()
    
    parallelization = {
        "algorithm_selection": 1,  # Disabled (heavy overhead)
        "fold_evaluation": 1,      # Disabled (light tasks, ProcessPool overhead too high)
        "threshold_testing": 1,    # Disabled (light tasks, ProcessPool overhead too high)
    }
    CONFIG["parallelization"] = parallelization
    
    # Log backtesting plan and parallelization
    app_reporter.log_backtesting_plan(n_strategies, n_thresholds)
    app_reporter.log_parallelization(n_algorithms, n_strategies, n_thresholds, parallelization, n_cpu)

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
        app_reporter.log_completion()
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise

"""TODOs"""

"""Need to create test files, too many things can break now"""
"""Why does mean reversion need threshold ? can it just not have ? adjust periods..."""
"""Undestand fully the strategies, create new ones"""
"""Put clearer parameter usage of strategies in README"""

"""Future testing"""

"""Test what is more effective, full deployment or cash fallbacks"""
"""Stocks selection, more than 5 causes overfitting, need to think of a way to get "similar" stocks to diversify"""
"""Test different algorithms (possible deep learning ? LSTM, CNN, GRU, XGBoost, etc.)"""
"""Explore more probabilities to chose best ML model, using strategy, etc. (not only IC)"""
"""Test more buy/sell strategies"""
"""Validation fine tunning, find right number of windows days..."""
"""Parameter tuning/more features ?"""
"""Possible signal decomposotion"""
"""Eigen portfolios pca ?"""
"""Encapsulating framework"""