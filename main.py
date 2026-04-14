from multiprocessing import cpu_count
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
    # Data parameters - 16 stocks
    # "tickers": [
    #     # Tech (5)
    #     "SPY", "AAPL", "MSFT", "GOOGL", "AMZN",
    #     # Financial (3)
    #     "JPM", "BAC", "GS",
    #     # Healthcare (3)
    #     "JNJ", "PFE", "UNH",
    #     # Energy (2)
    #     "XOM", "CVX",
    #     # Utilities (2)
    #     "NEE", "DUK",
    #     # Consumer (2)
    #     "WMT", "HD"
    # ],
    # testing with 5 stocks also... works better to reduce overfitting
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
    "probability_thresholds": [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60],  # Thresholds to test for position sizing
    "position_sizing": "probability_weighted",  # "equal_weight" or "probability_weighted"
    
    # Feature engineering parameters (placeholder for future extensions)
    "feature_profile": "classification_indicators",
    "lookback_period": 20,
    
    # ================== POSITION SELECTION ==================
    # NOTE: Backtest tests ALL 8 strategies and shows which performed best!
    #
    # Strategies tested automatically (no need to select one):
    #   "ensemble_smart"      - Consensus of 4 factors (RSI, momentum, mean reversion, vol)
    #   "momentum"            - Follows price trends
    #   "mean_reversion"      - Buys undervalued (price < SMA50)
    #   "volatility_weighted" - Reduces position size in high volatility
    #   "threshold"           - Baseline: probability threshold only
    #   "buy_and_hold"        - Benchmark: buy day 1, hold forever
    #   "fixed_income"        - Benchmark: 100% cash (conservative)
    #   "simple_reversal"     - Benchmark: SMA20 technical analysis only
    #
    # Position Selection (how many stocks to trade per day):
    #   "all"     - Trade all stocks beating threshold (diversified)
    #   "top_1"   - Trade only best stock of the day (concentrated)
    #   "top_3"   - Trade top 3 stocks (conservative)
    #   "top_5"   - Trade top 5 stocks (RECOMMENDED - balanced)
    #   "top_10"  - Trade top 10 stocks (diversified)
    # ============================================================================
    "position_selection": "top_5",           # How many stocks to select per day
    
    # ================== ALLOCATION MODE ==================
    # CHOOSE BETWEEN TWO PORTFOLIO STRATEGIES:
    #
    # 1. "cash_allocation" (CONSERVATIVE)
    #    - If model selects N stocks (N < all): invest in N, rest stays in CASH
    #    - Example: Model picks 2 of 5 stocks → 50% invested, 50% in CASH
    #    - Only goes 100% cash if NO stock beats purchase_threshold
    #    - ✓ Best for: Risk-averse, selective timing, avoid overexposure
    #    - Example use: "I only invest when I'm really confident"
    #
    # 2. "full_deployment" (AGGRESSIVE)
    #    - If model selects N stocks: invest 100% equally across those N
    #    - Example: Model picks 2 of 5 stocks → 50% each (100% deployed)
    #    - Only goes 100% cash if NO stock beats purchase_threshold
    #    - ✓ Best for: Capital efficiency, always deployed, avoids underutilization
    #    - Example use: "Deploy all capital to best opportunities, no idle cash"
    #
    # purchase_threshold: Minimum probability required to open ANY position
    #    - If best probability < purchase_threshold → 100% CASH (no position opened)
    #    - Only matters when NO selected stock beats this level
    #    - Example: purchase_threshold=0.52 → needs at least 52% confidence
    # ============================================================================
    "allocation_mode": "full_deployment",    # "cash_allocation" or "full_deployment"
    "purchase_threshold": 0.50,              # Minimum confidence to open any position
    
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
            "C": 10.0,  # Increased regularization to improve convergence
            "max_iter": 2000,  # Standard limit; convergence improved by higher C
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

"""Need to create test files, too many things can break now"""
"""Run with unity, try all possibilities!"""
"""Test what is more effective, full deployment or cash fallbacks | Make a better explanation of why it is useful"""

"""Good to have"""

"""Stocks selection, more than 5 causes overfitting, need to think of a way to get "similar" stocks to diversify"""
"""Test different algorithms (possible deep learning ? LSTM, CNN, GRU, XGBoost, etc.)"""
"""Explore more probabilities to chose best ML model, using strategy, etc. (not only IC)"""
"""Test more buy/sell strategies"""
"""Parameter tuning/more features ?"""
"""Clean requirements.txt"""