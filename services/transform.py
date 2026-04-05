import pandas as pd
import numpy as np

class FeatureEngineer:
    def build(self, raw_df: pd.DataFrame, feature_profile: str) -> tuple[pd.DataFrame, list[str], str]:
        if feature_profile == "xgboost":
            return self._build_xgboost_features(raw_df)

        raise NotImplementedError(
            f"Feature profile '{feature_profile}' is not implemented yet"
        )

    def _build_xgboost_features(self, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
        df = raw_df.copy().sort_values("date").reset_index(drop=True)

        df["return_1d"] = df["close"].pct_change()
        df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]
        df["oc_pct"] = (df["close"] - df["open"]) / df["open"]
        df["hl_pct"] = (df["high"] - df["low"]) / df["open"]

        for window in [3, 5, 10, 20, 60]:
            df[f"sma_{window}"] = df["close"].rolling(window).mean()
            df[f"std_{window}"] = df["close"].rolling(window).std()
            df[f"min_{window}"] = df["close"].rolling(window).min()
            df[f"max_{window}"] = df["close"].rolling(window).max()
            df[f"vol_mean_{window}"] = df["volume"].rolling(window).mean()
            df[f"vol_std_{window}"] = df["volume"].rolling(window).std()
            df[f"ret_mean_{window}"] = df["return_1d"].rolling(window).mean()
            df[f"ret_std_{window}"] = df["return_1d"].rolling(window).std()

        df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        df["volume_change"] = df["volume"].pct_change()
        df["obv"] = np.where(
            df["close"] > df["close"].shift(1),
            df["volume"],
            np.where(df["close"] < df["close"].shift(1), -df["volume"], 0.0),
        ).cumsum()

        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter

        df["target_return_next_day"] = df["close"].shift(-1) / df["close"] - 1

        feature_cols = [
            "open", "high", "low", "close", "adj_close", "volume",
            "return_1d", "log_return_1d", "range_pct", "oc_pct", "hl_pct",
            "sma_3", "sma_5", "sma_10", "sma_20", "sma_60",
            "std_3", "std_5", "std_10", "std_20", "std_60",
            "min_5", "max_5", "min_20", "max_20",
            "vol_mean_5", "vol_mean_20", "vol_mean_60",
            "vol_std_5", "vol_std_20", "vol_std_60",
            "ret_mean_5", "ret_mean_20", "ret_mean_60",
            "ret_std_5", "ret_std_20", "ret_std_60",
            "ema_5", "ema_12", "ema_26", "macd", "macd_signal", "macd_hist",
            "rsi_14", "volume_change", "obv", "day_of_week", "month", "quarter",
        ]

        dataset = df[["date"] + feature_cols + ["target_return_next_day"]].copy()
        dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        return dataset, feature_cols, "target_return_next_day"