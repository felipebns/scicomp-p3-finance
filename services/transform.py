import pandas as pd

class FeatureEngineer:
    def build(self, raw_df: pd.DataFrame, feature_profile: str) -> tuple[pd.DataFrame, list[str], str]:
        if feature_profile == "classification_indicators":
            return self._build_classification_indicators(raw_df)
        raise NotImplementedError(f"Feature profile '{feature_profile}' is not implemented yet")

    def _build_classification_indicators(self, df: pd.DataFrame, lookback: int = 20) -> tuple[pd.DataFrame, list[str], str]:
        df = df.copy()
        target_col = "target_direction"
        df[target_col] = (df["close"].shift(-1) > df["close"]).astype(int)

        features = []

        # 1. EMA
        df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        features.extend(["ema_10", "ema_20"])

        # 2. RSI (14 periods)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"].fillna(50)
        features.append("rsi_14")

        # 3. Bollinger Bands (20 periods)
        sma = df["close"].rolling(window=20).mean()
        std = df["close"].rolling(window=20).std()
        df["bb_upper"] = sma + (2 * std)
        df["bb_lower"] = sma - (2 * std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
        features.extend(["bb_upper", "bb_lower", "bb_width"])

        # 4. ROC (Rate of Change) - 10 periods
        df["roc_10"] = ((df["close"] - df["close"].shift(10)) / df["close"].shift(10)) * 100
        features.append("roc_10")

        # 5. Momentum - 10 periods
        df["momentum_10"] = df["close"] - df["close"].shift(10)
        features.append("momentum_10")

        # 6. Returns and Lagged Features
        df["return"] = df["close"].pct_change()
        features.append("return")

        for i in range(1, 6): # Last 5 days of lagged returns
            col = f"return_lag_{i}"
            df[col] = df["return"].shift(i)
            features.append(col)

        # Drop rows with NaN due to rolling/shift windows, except the very last row for target
        # We need the target to be NA for the last row (we can't know the future), 
        # but we drop NAs where features are missing
        df = df.dropna(subset=features).copy()
        
        # Drop the last row because its target is invalid (future data)
        df = df.iloc[:-1].reset_index(drop=True)

        return df, features, target_col