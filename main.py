from services.pipeline import Pipeline
from services.stock import Stock
from services.xgboost import XGBoostAlgorithm
from services.lstm import LstmAlgorithm
from services.gru import GruAlgorithm
from services.hyperparameter_sweep import HyperparameterSweep
from datetime import datetime, timezone

if __name__ == "__main__":
    ticker = "AAPL"
    start = "2016-01-01"
    end = datetime.now(timezone.utc).date().isoformat()

    # Define underlying stock fetcher
    stock = Stock(ticker=ticker, start=start, end=end)
    
    # Define Parameter Grid for GRU Sweep
    param_grid = {
        'lookback': [10, 15, 20],
        'hidden_size': [32, 64],
        'num_layers': [1, 2],
        'batch_size': [32, 64],
        'epochs': [100],
        'patience': [15]
    }
    
    # 1. Run hyperparameter sweep using the Pipeline internally
    sweep = HyperparameterSweep(
        algorithm_class=GruAlgorithm,
        stock=stock,
        param_grid=param_grid,
        output_path="output/gru_sweep.json",
        metric_to_optimize="r2"  # Optimize for R² score
    )
    
    sweep.run(verbose=True)
    sweep.save_results()
    
    # 2. Extract best algorithm from the sweep
    print("\n============================================================")
    print("TRAINING FINAL MODEL WITH BEST SWEEP CONFIGURATION")
    print("============================================================")
    
    best_algorithm = GruAlgorithm(**sweep.best_config)
    
    # 3. Create single Pipeline using the optimal algorithm and run full output
    final_pipeline = Pipeline(stock=stock, algorithm=best_algorithm, output_dir="output")
    metrics = final_pipeline.run(save=True)

"""Try latter with log returns instead of absolute returns, maybe more stationary and easier to predict"""

"""Change target ? up or down ? return 5 days ?"""

"""Feature optimization is needed, changing number of layers, hidden units, rolling window, etc"""

"""More features in RNN"""

"""Sweep with different algorithms, not just GRU, but also LSTM and XGBoost, to compare performance"""

"""What should I maximize ? r2 ? rmse ? mae ?"""

"""Better parameters for sweeping"""

"""Change stopping criteria, maybe more patience ?"""