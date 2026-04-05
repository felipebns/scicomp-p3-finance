from services.pipeline import Pipeline
from services.stock import Stock
from services.xgboost import XGBoostAlgorithm
from datetime import datetime, timezone

if __name__ == "__main__":
    ticker = "AAPL"
    start = "2016-01-01"
    end = datetime.now(timezone.utc).date().isoformat()

    stock = Stock(ticker=ticker, start=start, end=end)
    algorithm = XGBoostAlgorithm(random_state=42)
    pipeline = Pipeline(stock=stock, algorithm=algorithm, output_dir="output")
    pipeline.run()
