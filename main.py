from services.pipeline import Pipeline
from services.stock import Stock
from services.xgboost import XGBoostAlgorithm
from services.gru import GruAlgorithm 
from services.lstm import LstmAlgorithm
from datetime import datetime, timezone

if __name__ == "__main__":
    ticker = "AAPL"
    start = "2016-01-01"
    end = datetime.now(timezone.utc).date().isoformat()

    stock = Stock(ticker=ticker, start=start, end=end)
    # algorithm = XGBoostAlgorithm(random_state=42)
    algorithm = GruAlgorithm(lookback=20, random_state=42, epochs=100, batch_size=64)
    # algorithm = LstmAlgorithm(lookback=10, random_state=42, epochs=100, batch_size=64)
    pipeline = Pipeline(stock=stock, algorithm=algorithm, output_dir="output")
    pipeline.run()

"""Need to change strategy ? bad results"""

"""Try latter with log returns instead of absolute returns, maybe more stationary and easier to predict"""

"""Lstm and gru are very similar, can I just have one class with a parameter to choose between them?"""

"""Parameter/feature optimization is needed"""