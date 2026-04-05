import numpy as np
import pandas as pd
import yfinance as yf

class Stock:
    def __init__(self, ticker: str, start: str, end: str) -> None:
        self.ticker = ticker
        self.start = start
        self.end = end

    def fetch(self) -> pd.DataFrame:
        try:
            df = yf.download(
                tickers=self.ticker,
                start=self.start,
                end=self.end,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )
            return self._normalize(df)
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            return

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[0]) for col in df.columns]

        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        if "date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                idx_name = df.index.name or "index"
                df = df.reset_index().rename(columns={idx_name: "date"})
            else:
                df = df.reset_index().rename(columns={df.columns[0]: "date"})

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = np.nan

        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]

        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"]).reset_index(drop=True)
        return df[["date", "open", "high", "low", "close", "adj_close", "volume"]]