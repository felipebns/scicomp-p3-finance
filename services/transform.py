import pandas as pd
import numpy as np

class FeatureEngineer:
    def build(self, raw_df: pd.DataFrame, feature_profile: str) -> tuple[pd.DataFrame, list[str], str]:
        if feature_profile == "xgboost":
            return self._build_xgboost_features(raw_df)

        raise NotImplementedError(f"Feature profile '{feature_profile}' is not implemented yet")

    def _build_xgboost_features(self, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
        df = raw_df.copy().sort_values("date").reset_index(drop=True)

        df["adj_return_1d"] = df["adj_close"].pct_change()
        df["adj_log_return_1d"] = np.log(df["adj_close"] / df["adj_close"].shift(1))

        df["adj_range_pct"] = (df["adj_high"] - df["adj_low"]) / df["adj_close"]
        df["adj_oc_pct"] = (df["adj_close"] - df["adj_open"]) / df["adj_open"]
        df["adj_hl_pct"] = (df["adj_high"] - df["adj_low"]) / df["adj_open"]

        df["ema_9"] = df["adj_close"].ewm(span=9, adjust=False).mean()

        df["sma_5"] = df["adj_close"].rolling(5).mean()
        df["sma_15"] = df["adj_close"].rolling(15).mean()
        df["sma_30"] = df["adj_close"].rolling(30).mean()

        delta = df["adj_close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        ema_12 = df["adj_close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["adj_close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        df["volume_change"] = df["volume"].pct_change()

        direction = np.sign(df["adj_close"].diff()).fillna(0.0)
        df["obv"] = (direction * df["volume"]).cumsum()

        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter

        df["target_return_next_day"] = df["adj_close"].shift(-1) / df["adj_close"] - 1

        # Only use everything adjusted!
        feature_cols = [
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "volume",
            "adj_return_1d",
            "adj_log_return_1d",
            "adj_range_pct",
            "adj_oc_pct",
            "adj_hl_pct",
            "ema_9",
            "sma_5",
            "sma_15",
            "sma_30",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "volume_change",
            "obv",
            "day_of_week",
            "month",
            "quarter",
        ]

        dataset = df[["date"] + feature_cols + ["target_return_next_day"]].copy()
        dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        return dataset, feature_cols, "target_return_next_day"