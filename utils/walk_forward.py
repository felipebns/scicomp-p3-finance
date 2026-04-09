import pandas as pd
import numpy as np
from typing import Dict

def walk_forward_validation(df: pd.DataFrame, algorithm, features: list[str], target_col: str, 
                            train_window: int = 1000, test_window: int = 250) -> Dict:
    """
    Perform rigorous Walk-Forward Validation (WFV) without data leakage.
    - GROWING WINDOW: Always train on [0:train_end], test on [train_end:test_end]
    - Each iteration rolls BOTH boundaries forward by test_window
    - No overlap or reuse of test data in training
    
    Example with train_window=750, test_window=250:
      Iter 0: train[0:750],     test[750:1000]
      Iter 1: train[0:1000],    test[1000:1250]
      Iter 2: train[0:1250],    test[1250:1500]
    """
    all_predictions = []
    all_test_indices = []
    
    n = len(df)
    train_end = train_window
    test_end = train_window + test_window
    
    while test_end <= n:
        train_df = df.iloc[:train_end]  # GROWING: all data up to this point
        test_df = df.iloc[train_end:test_end]
        
        # Train model on historical data
        algorithm.fit(train_df, pd.DataFrame(), features, target_col)
        
        # Predict on future (blind) test period
        predictions = algorithm.predict(test_df, features)
        
        all_predictions.extend(predictions)
        all_test_indices.extend(test_df.index)
        
        # Roll forward for next iteration (no overlap)
        train_end = test_end
        test_end = train_end + test_window
        
    # Reconstruct aggregated OOS dataframe and predictions
    wfv_df = df.loc[all_test_indices].copy()
    wfv_predictions = np.array(all_predictions)
    
    return wfv_df, wfv_predictions