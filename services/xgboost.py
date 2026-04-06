from xgboost import XGBRegressor
from services.algorithm import Algorithm
import numpy as np
import pandas as pd

class XGBoostAlgorithm(Algorithm):
    def __init__(self, n_estimators: int = 400, max_depth: int = 8, learning_rate: float = 0.01,
                gamma: float = 0.01, subsample: float = 0.8, colsample_bytree: float = 0.8,
                reg_alpha: float = 0.0, reg_lambda: float = 1.0, min_child_weight: int = 2,
                random_state: int = 42) -> None:
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str], target_col: str) -> None:
        self.model.fit(
            train_df[features],
            train_df[target_col],
            eval_set=[(valid_df[features], valid_df[target_col])],
            verbose=False,
        )

    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        return self.model.predict(df[features])

    def name(self) -> str:
        return "xgboost"

    def feature_profile(self) -> str:
        return "xgboost"