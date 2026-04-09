from datetime import datetime, timezone
from services.stock import Stock
from services.pipeline import Pipeline
from services.algorithms.svc import SVCAlgorithm
from services.algorithms.random_forest import RandomForestAlgorithm
from services.algorithms.ensemble import EnsembleClassificationAlgorithm
from services.algorithms.logistic_regression import LogisticRegressionAlgorithm

if __name__ == "__main__":
    ticker = "SPY"
    start = "2006-01-01" #20 years of data
    end = datetime.now(timezone.utc).date().isoformat()

    print(f"Fetching historical data for {ticker} since {start}...")
    stock = Stock(ticker=ticker, start=start, end=end)

    models = [
        LogisticRegressionAlgorithm(),
        SVCAlgorithm(),
        RandomForestAlgorithm(),
        EnsembleClassificationAlgorithm()
    ]

    print("\n============================================================")
    print("STARTING CLASSIFICATION PIPELINE WITH ENSEMBLE LEARNING")
    print("============================================================")

    pipeline = Pipeline(
        stock=stock, 
        algorithms=models, 
        output_dir="output",
        test_size=0.20,
        history_window=100
    )

    results = pipeline.run(save=True)

    print("\n============================================================")
    print("PIPELINE COMPLETED. RESULTS SAVED IN 'output/' DIRECTORY")
    print("============================================================")

"""Test different algorithms (possible deep learning ? LSTM, CNN, GRU, XGBoost, etc.)"""
"""Manage multiple stocks, weighting, portfolio-level metrics"""
"""Parameter tuning/more features ?"""
"""Test different buy/sell strategies, what is happening in sharpe strategy and _calculate_strategy_returns ?"""



"""Clean requirements"""

"""1 Model per stock for now !!!"""