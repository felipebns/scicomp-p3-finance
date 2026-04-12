from datetime import datetime, timezone
from services.stock import Stock
from services.pipeline import Pipeline
from services.algorithms.svc import SVCAlgorithm
from services.algorithms.random_forest import RandomForestAlgorithm
from services.algorithms.ensemble import EnsembleClassificationAlgorithm
from services.algorithms.logistic_regression import LogisticRegressionAlgorithm
from services.logger_config import setup_logging, get_logger

# ============================================================
# CONFIGURATION: All parameters can be modified here
# ============================================================

CONFIG = {
    # Data parameters
    "tickers": ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"],
    "start_date": "2006-01-01",
    
    # Pipeline parameters
    "output_dir": "output",
    "test_size": 0.20,
    
    # Walk-Forward Validation parameters
    "wfv_train_window": 750,    # ~3 years of training data
    "wfv_test_window": 250,     # ~1 year of validation data
    
    # Backtesting parameters
    "initial_capital": 10000,
    "transaction_cost": 0.0005,  # 0.05% per trade
    "slippage": 0.0005,          # 0.05% slippage
    "annual_rf_rate": 0.05,      # 5% annual risk-free rate
    "probability_thresholds": [0.50, 0.52, 0.53, 0.55, 0.57],
    "position_sizing": "probability_weighted",  # "equal_weight" or "probability_weighted"
    
    # Feature engineering parameters (placeholder for future extensions)
    "feature_profile": "classification_indicators",
    "lookback_period": 20,
    
    # Model hyperparameters
    "model_params": {
        "LogisticRegression": {
            "random_state": 42,
            "max_iter": 1000,
            "class_weight": "balanced",
            "C": 1.0,
        },
        "SVC": {
            "kernel": "linear",
            "random_state": 42,
            "probability": True,
            "class_weight": "balanced",
            "C": 1.0,
            "max_iter": 2000,
        },
        "RandomForest": {
            "n_estimators": 100,
            "random_state": 42,
            "class_weight": "balanced",
            "max_depth": None,
            "n_jobs": -1,
        },
        "Ensemble": {
            # Ensemble uses sub-models, so individual params not directly used
            "voting": "soft",
        }
    }
}

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
    logger.info(f"  Tickers: {CONFIG['tickers']}")
    logger.info(f"  Position Sizing: {CONFIG['position_sizing']}")
    logger.info(f"  Transaction Cost: {CONFIG['transaction_cost']:.4%}")
    logger.info(f"  Initial Capital: ${CONFIG['initial_capital']}")
    logger.info(f"  WFV Windows: train={CONFIG['wfv_train_window']}, test={CONFIG['wfv_test_window']}")
    logger.info(f"  Probability Thresholds: {CONFIG['probability_thresholds']}")

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
    )

    try:
        results = pipeline.run(save=True)
        logger.info("All results saved to 'output/' and 'logs/' directories")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise

"""Test different algorithms (possible deep learning ? LSTM, CNN, GRU, XGBoost, etc.)"""
"""Parameter tuning/more features ?"""
"""Slow, change to polars ? new framework ? | Parallelize WFV ? | Use more efficient backtesting framework ? | Pickle files ?"""

"""Add more stocks"""
"""Explore more probabilities to chose best ML model, using strategy, etc. (not only IC)"""
"""Test more buy/sell strategies"""
"""Clean requirements.txt"""