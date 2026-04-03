"""Daily stock prediction pipeline using yfinance -> Alpha Vantage fallback, Parquet storage, and XGBoost.

What this script does:
1) Tries to download daily OHLCV data from yfinance.
2) If that fails, tries Alpha Vantage (requires ALPHAVANTAGE_API_KEY).
3) If both fail, it exits without training.
4) Stores raw and feature-engineered datasets as Parquet.
5) Trains a single XGBoost regressor to predict next-day close.
6) Saves the model and prints the next-day prediction.

Notes:
- This is intentionally a single-file starter pipeline.
- It uses time-based splitting (no shuffling).
- Feature engineering is deliberately compact but practical.
- You can later swap the target to next-day return if you prefer.

Environment variables:
- ALPHAVANTAGE_API_KEY: required only if yfinance fails.

Example usage:
    python main.py --ticker AAPL --start 2015-01-01 --outdir data

Dependencies:
    pip install pandas numpy pyarrow scikit-learn xgboost yfinance requests joblib
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("portfolio-pipeline")


@dataclass
class PipelineConfig:
    ticker: str
    start: str
    end: Optional[str]
    outdir: Path
    test_size: float = 0.15
    valid_size: float = 0.15
    random_state: int = 42
    min_rows: int = 120


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Daily stock prediction pipeline")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--outdir", default="./artifacts", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split fraction")
    parser.add_argument("--valid-size", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--min-rows", type=int, default=120, help="Minimum rows required to train")
    args = parser.parse_args()
    return PipelineConfig(
        ticker=args.ticker.upper().strip(),
        start=args.start,
        end=args.end,
        outdir=Path(args.outdir),
        test_size=args.test_size,
        valid_size=args.valid_size,
        random_state=args.random_state,
        min_rows=args.min_rows,
    )


def ensure_output_dirs(outdir: Path) -> dict[str, Path]:
    raw_dir = outdir / "raw"
    processed_dir = outdir / "processed"
    models_dir = outdir / "models"
    reports_dir = outdir / "reports"
    for d in (raw_dir, processed_dir, models_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "raw": raw_dir,
        "processed": processed_dir,
        "models": models_dir,
        "reports": reports_dir,
    }


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    # yfinance sometimes returns a MultiIndex like ('Close', 'AAPL').
    # In that case, flatten to the first level, which contains the OHLCV names.
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            # Prefer the field name (e.g. 'Close') over the ticker level.
            if len(col) >= 1 and str(col[0]).strip():
                flat_cols.append(str(col[0]))
            else:
                flat_cols.append(str(col[-1]))
        df.columns = flat_cols

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Common yfinance column names after flattening/lowercasing.
    rename_map = {
        "adj_close": "adj_close",
        "adjusted_close": "adj_close",
        "close": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            idx_name = df.index.name or "index"
            df = df.reset_index().rename(columns={idx_name: "date"})
        else:
            df = df.reset_index().rename(columns={df.columns[0]: "date"})

    df["date"] = pd.to_datetime(df["date"], utc=False)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # Ensure standard OHLCV columns exist.
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan

    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    numeric_cols = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df[[c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]]


def fetch_yfinance(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed")

    logger.info("Trying yfinance for %s", ticker)
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data")

    return _normalize_ohlcv(df)


def fetch_alpha_vantage(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not set")

    logger.info("Trying Alpha Vantage for %s", ticker)
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": api_key,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if "Error Message" in payload:
        raise RuntimeError(f"Alpha Vantage error: {payload['Error Message']}")
    if "Note" in payload:
        raise RuntimeError(f"Alpha Vantage rate limit or note: {payload['Note']}")

    ts_key = "Time Series (Daily)"
    if ts_key not in payload:
        raise RuntimeError(f"Alpha Vantage unexpected response keys: {list(payload.keys())}")

    rows = []
    for date_str, values in payload[ts_key].items():
        rows.append(
            {
                "date": pd.to_datetime(date_str),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "adj_close": float(values.get("5. adjusted close", values["4. close"])),
                "volume": float(values["6. volume"]),
            }
        )
    df = pd.DataFrame(rows)
    df = _normalize_ohlcv(df)

    start_ts = pd.to_datetime(start)
    if end is None:
        end_ts = pd.Timestamp.now().normalize()
    else:
        end_ts = pd.to_datetime(end)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("Alpha Vantage returned no rows in the requested date range")

    return df


def fetch_market_data(ticker: str, start: str, end: Optional[str]) -> Tuple[pd.DataFrame, str]:
    errors: list[str] = []

    try:
        df = fetch_yfinance(ticker, start, end)
        return df, "yfinance"
    except Exception as exc:
        errors.append(f"yfinance: {exc}")
        logger.warning("yfinance failed: %s", exc)

    try:
        df = fetch_alpha_vantage(ticker, start, end)
        return df, "alpha_vantage"
    except Exception as exc:
        errors.append(f"alpha_vantage: {exc}")
        logger.warning("Alpha Vantage failed: %s", exc)

    raise RuntimeError("Both data sources failed: " + " | ".join(errors))


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Price-derived features
    df["return_1d"] = df["close"].pct_change()
    df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["oc_pct"] = (df["close"] - df["open"]) / df["open"]
    df["hl_pct"] = (df["high"] - df["low"]) / df["open"]

    # Rolling statistics
    for window in (3, 5, 10, 20, 60):
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"std_{window}"] = df["close"].rolling(window).std()
        df[f"min_{window}"] = df["close"].rolling(window).min()
        df[f"max_{window}"] = df["close"].rolling(window).max()
        df[f"vol_mean_{window}"] = df["volume"].rolling(window).mean()
        df[f"vol_std_{window}"] = df["volume"].rolling(window).std()
        df[f"ret_mean_{window}"] = df["return_1d"].rolling(window).mean()
        df[f"ret_std_{window}"] = df["return_1d"].rolling(window).std()

    # Exponential moving averages
    for span in (5, 12, 26):
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()

    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # RSI (14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Volume features
    df["volume_change"] = df["volume"].pct_change()
    df["obv"] = np.where(df["close"] > df["close"].shift(1), df["volume"], np.where(df["close"] < df["close"].shift(1), -df["volume"], 0.0)).cumsum()

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Target: next-day close (shifted by -1)
    df["target_close_next_day"] = df["close"].shift(-1)

    # Prediction-time helper: close-to-next-close delta
    df["target_return_next_day"] = df["target_close_next_day"] / df["close"] - 1

    return df


def build_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str], str]:
    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "return_1d",
        "log_return_1d",
        "range_pct",
        "oc_pct",
        "hl_pct",
        "sma_3",
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_60",
        "std_3",
        "std_5",
        "std_10",
        "std_20",
        "std_60",
        "min_5",
        "max_5",
        "min_20",
        "max_20",
        "vol_mean_5",
        "vol_mean_20",
        "vol_mean_60",
        "vol_std_5",
        "vol_std_20",
        "vol_std_60",
        "ret_mean_5",
        "ret_mean_20",
        "ret_mean_60",
        "ret_std_5",
        "ret_std_20",
        "ret_std_60",
        "ema_5",
        "ema_12",
        "ema_26",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi_14",
        "volume_change",
        "obv",
        "day_of_week",
        "month",
        "quarter",
    ]
    # Use next-day RETURN as target (more stationary than price)
    target_col = "target_return_next_day"

    available = [c for c in feature_cols if c in df.columns]
    dataset = df[["date"] + available + [target_col]].copy()
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna().reset_index(drop=True)
    return dataset, available, target_col


def time_split(df: pd.DataFrame, test_size: float, valid_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1 or not 0 < valid_size < 1:
        raise ValueError("test_size and valid_size must be between 0 and 1")
    if test_size + valid_size >= 1:
        raise ValueError("test_size + valid_size must be less than 1")

    n = len(df)
    train_end = int(n * (1 - valid_size - test_size))
    valid_end = int(n * (1 - test_size))

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()
    return train, valid, test


def xgb_params(random_state: int) -> dict:
    # A sensible baseline for daily stock regression.
    return {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "min_child_weight": 2,
        "gamma": 0.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
    }


def train_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str], target_col: str, random_state: int) -> XGBRegressor:
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_valid = valid_df[features]
    y_valid = valid_df[target_col]

    model = XGBRegressor(**xgb_params(random_state))
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    return model


def evaluate(model: XGBRegressor, df: pd.DataFrame, features: list[str], target_col: str) -> dict:
    X = df[features]
    y_true = df[target_col].to_numpy()
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to Parquet, ensuring the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_metrics(metrics: dict, path: Path) -> None:
    """Write metrics to JSON with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def evaluate_against_random_walk(model: XGBRegressor, df: pd.DataFrame, features: list[str], target_col: str) -> dict:
    """Compare the model against a next-day random walk baseline.

    Since the target is next-day return, the random walk baseline is return = 0.
    """
    X = df[features]
    y_true = df[target_col].to_numpy()
    y_pred = model.predict(X)
    y_rw = np.zeros_like(y_true, dtype=float)

    return {
        "model": {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        },
        "random_walk": {
            "mae": float(mean_absolute_error(y_true, y_rw)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_rw))),
            "r2": float(r2_score(y_true, y_rw)),
        },
    }


def plot_predictions_vs_random_walk(
    df: pd.DataFrame,
    model: XGBRegressor,
    features: list[str],
    output_path: Path,
    history_window: int = 250,
) -> None:
    """Plot RETURNS instead of prices (more aligned with the target).

    Shows:
    - Actual next-day return
    - XGBoost predicted return
    - Random walk (0 return)
    """
    if len(df) < 2:
        return

    working = df.copy().reset_index(drop=True)
    X = working[features]
    pred_return = model.predict(X)

    actual_return = working["target_return_next_day"]
    dates = working["date"]

    plot_df = pd.DataFrame(
        {
            "date": dates,
            "actual_return": actual_return,
            "model_return": pred_return,
            "random_walk": np.zeros_like(pred_return),
        }
    ).dropna().reset_index(drop=True)

    if len(plot_df) > history_window:
        plot_df = plot_df.iloc[-history_window:].copy()

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(plot_df["date"], plot_df["actual_return"], label="Actual return")
    ax.plot(plot_df["date"], plot_df["model_return"], label="XGBoost prediction")
    ax.plot(plot_df["date"], plot_df["random_walk"], label="Random walk (0)")

    # Highlight last point
    last = plot_df.iloc[-1]
    last_date = last["date"]

    for label, col, color in [
        ("Actual", "actual_return", "tab:blue"),
        ("Model", "model_return", "tab:green"),
        ("RW", "random_walk", "tab:red"),
    ]:
        val = last[col]
        ax.scatter([last_date], [val], color=color, s=40)
        ax.annotate(f"{label}: {val*100:.2f}%",
                    xy=(last_date, val),
                    xytext=(last_date + pd.Timedelta(days=2), val),
                    arrowprops=dict(arrowstyle="-", color=color),
                    fontsize=9,
                    color=color)

    ax.set_title("Return Prediction vs Random Walk")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    cfg = parse_args()
    dirs = ensure_output_dirs(cfg.outdir)

    end = cfg.end
    if end is None:
        end = datetime.now(timezone.utc).date().isoformat()

    logger.info("Starting pipeline for %s", cfg.ticker)

    try:
        raw_df, source = fetch_market_data(cfg.ticker, cfg.start, end)
    except Exception as exc:
        logger.error("Data ingestion failed. Model will not run. Reason: %s", exc)
        return 1

    logger.info("Data source used: %s | rows=%d", source, len(raw_df))
    raw_path = dirs["raw"] / f"{cfg.ticker}_{source}_raw.parquet"
    save_parquet(raw_df, raw_path)
    logger.info("Saved raw data to %s", raw_path)

    featured = add_technical_indicators(raw_df)
    dataset, feature_cols, target_col = build_dataset(featured)

    if len(dataset) < cfg.min_rows:
        logger.error(
            "Not enough rows after feature engineering to train. Have %d, need at least %d.",
            len(dataset),
            cfg.min_rows,
        )
        return 1

    processed_path = dirs["processed"] / f"{cfg.ticker}_featured.parquet"
    save_parquet(dataset, processed_path)
    logger.info("Saved processed data to %s", processed_path)

    train_df, valid_df, test_df = time_split(dataset, cfg.test_size, cfg.valid_size)
    logger.info(
        "Split sizes -> train=%d valid=%d test=%d",
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    model = train_model(train_df, valid_df, feature_cols, target_col, cfg.random_state)

    val_metrics = evaluate(model, valid_df, feature_cols, target_col)
    test_metrics = evaluate(model, test_df, feature_cols, target_col)
    baseline_comparison = evaluate_against_random_walk(model, test_df, feature_cols, target_col)

    model_path = dirs["models"] / f"{cfg.ticker}_xgboost.joblib"
    dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    latest_row = dataset.iloc[[-1]].copy()
    # Predict next-day RETURN directly
    next_return_pred = float(model.predict(latest_row[feature_cols])[0])
    last_close = float(latest_row["close"].iloc[0])
    # Convert predicted return back to price for convenience
    next_close_pred = last_close * (1.0 + next_return_pred)

    metrics = {
        "ticker": cfg.ticker,
        "source": source,
        "rows_raw": int(len(raw_df)),
        "rows_featured": int(len(dataset)),
        "features": feature_cols,
        "validation": val_metrics,
        "test": test_metrics,
        "baseline_comparison_test": baseline_comparison,
        "last_close": last_close,
        "predicted_next_close": next_close_pred,
        "predicted_next_return": next_return_pred,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    plot_path = dirs["reports"] / f"{cfg.ticker}_prediction_vs_random_walk.png"
    plot_predictions_vs_random_walk(test_df, model, feature_cols, plot_path)

    metrics_path = dirs["reports"] / f"{cfg.ticker}_metrics.json"
    write_metrics(metrics, metrics_path)

    logger.info("Validation metrics: %s", val_metrics)
    logger.info("Test metrics: %s", test_metrics)
    logger.info("Baseline comparison on test set: %s", baseline_comparison)
    logger.info("Last close: %.4f | Predicted next close: %.4f | Predicted next return: %.4f%%", last_close, next_close_pred, next_return_pred * 100)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved plot to %s", plot_path)

    # Tiny human-friendly output at the end.
    print("\n=== Prediction Result ===")
    print(f"Ticker: {cfg.ticker}")
    print(f"Data source used: {source}")
    print(f"Last close: {last_close:.4f}")
    print(f"Predicted next close: {next_close_pred:.4f}")
    print(f"Predicted next return: {next_return_pred * 100:.3f}%")
    print("\n=== Test Metrics ===")
    print(json.dumps(test_metrics, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
