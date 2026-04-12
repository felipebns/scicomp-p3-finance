import pandas as pd

class FeatureEngineer:
    def build(self, raw_df: pd.DataFrame, feature_profile: str) -> tuple[pd.DataFrame, list[str], str]:
        if feature_profile == "classification_indicators":
            return self._build_classification_indicators(raw_df)
        raise NotImplementedError(f"Feature profile '{feature_profile}' is not implemented yet")

    def _build_classification_indicators(self, df: pd.DataFrame, lookback: int = 20) -> tuple[pd.DataFrame, list[str], str]:
        df = df.copy()
        
        # Sort by ticker and date just in case
        df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)
        
        target_col = "target_direction"
        # Group by ticker for features
        df[target_col] = (df.groupby("ticker")["close"].shift(-1) > df["close"]).astype(int)

        features = []

        # 1. EMA
        df["ema_10"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=10, adjust=False).mean())
        df["ema_20"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
        features.extend(["ema_10", "ema_20"])

        # 2. RSI (14 periods)
        delta = df.groupby("ticker")["close"].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        # Groupby rolling can be tricky, let's just do it cleanly
        df["gain_roll"] = gain.groupby(df["ticker"]).transform(lambda x: x.rolling(window=14).mean())
        df["loss_roll"] = loss.groupby(df["ticker"]).transform(lambda x: x.rolling(window=14).mean())
        rs = df["gain_roll"] / df["loss_roll"]
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"].fillna(50)
        df.drop(columns=["gain_roll", "loss_roll"], inplace=True)
        features.append("rsi_14")

        # 3. Bollinger Bands (20 periods)
        sma = df.groupby("ticker")["close"].transform(lambda x: x.rolling(window=20).mean())
        std = df.groupby("ticker")["close"].transform(lambda x: x.rolling(window=20).std())
        df["bb_upper"] = sma + (2 * std)
        df["bb_lower"] = sma - (2 * std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
        features.extend(["bb_upper", "bb_lower", "bb_width"])

        # 4. ROC (Rate of Change) - 10 periods
        shifted_10 = df.groupby("ticker")["close"].shift(10)
        df["roc_10"] = ((df["close"] - shifted_10) / shifted_10) * 100
        features.append("roc_10")

        # 5. Momentum - 10 periods
        df["momentum_10"] = df["close"] - shifted_10
        features.append("momentum_10")

        # 6. Returns and Lagged Features
        df["return"] = df.groupby("ticker")["close"].pct_change()
        features.append("return")

        for i in range(1, 6): # Last 5 days of lagged returns
            col = f"return_lag_{i}"
            df[col] = df.groupby("ticker")["return"].shift(i)
            features.append(col)

        # Precompute next return for safe evaluation later
        df["next_return"] = df.groupby("ticker")["return"].shift(-1)

        # Drop rows with NaN due to rolling/shift windows, except the very last row for target
        df = df.dropna(subset=features).copy()
        
        # Drop the last row of each ticker because its target is invalid (future data)
        df = df.drop(df.groupby("ticker").tail(1).index).reset_index(drop=True)
        
        # Sort by date then ticker so time-based splits and backtests work natively
        df = df.sort_values(by=["date", "ticker"]).reset_index(drop=True)

        return df, features, target_col