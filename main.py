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

    print(f"Buscando dados históricos de {ticker} desde {start}...")
    stock = Stock(ticker=ticker, start=start, end=end)

    models = [
        LogisticRegressionAlgorithm(),
        SVCAlgorithm(),
        RandomForestAlgorithm(),
        EnsembleClassificationAlgorithm()
    ]

    print("\n============================================================")
    print("INICIANDO PIPELINE DE CLASSIFICAÇÃO COM ENSEMBLE LEARNING")
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
    print("PIPELINE CONCLUÍDO. RESULTADOS SALVOS NO DIRETÓRIO 'output/'")
    print("============================================================")

"""Manage multiple stocks, weighting, portfolio-level metrics"""
"""Test different sharpe strategies, should I choose the one with the highest sharpe ratio?"""
"""Test different buy/sell strategies"""
"""Test different algorithms (possible deep learning ? LSTM, CNN, GRU, XGBoost, etc.)"""
"""Parameter tuning"""
"""Verify conceptual correctness of everything, anualized return, sharpe ratio, drawdown, etc."""
"""Tirar portuges depois, deixar só inglês"""