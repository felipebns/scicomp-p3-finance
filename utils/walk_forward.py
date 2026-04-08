import pandas as pd
import numpy as np
from typing import Dict

def walk_forward_validation(df: pd.DataFrame, algorithm, features: list[str], target_col: str, 
                            train_window: int = 1000, test_window: int = 250) -> Dict:
    """
    Perform rigorous Walk-Forward Validation (WFV) without data leakage.
    - Trains on `train_window` trading days.
    - Tests on subsequent `test_window` days.
    - Rolls the window forward and repeats.
    """
    all_predictions = []
    all_test_indices = []
    
    n = len(df)
    for start_idx in range(0, n - train_window - test_window, test_window):
        train_end = start_idx + train_window
        test_end = train_end + test_window
        
        train_df = df.iloc[start_idx:train_end]
        test_df = df.iloc[train_end:test_end]
        
        # Treina o modelo apenas com o passado preenchendo as features necessárias
        algorithm.fit(train_df, pd.DataFrame(), features, target_col)
        
        # Avalia no futuro cego
        predictions = algorithm.predict(test_df, features)
        
        all_predictions.extend(predictions)
        all_test_indices.extend(test_df.index)
        
    # Reconstroi o dataframe de teste agregado
    wfv_df = df.loc[all_test_indices].copy()
    wfv_predictions = np.array(all_predictions)
    
    return wfv_df, wfv_predictions