import pandas as pd
import numpy as np
from services.log.logger_config import get_logger

logger = get_logger()

class PositionNormalizer:
    """Normalizes position vectors to capital fractions.
    
    Pipeline:
    1. GATE: Check if best_signal meets purchase_threshold (confidence gate)
       -> If NO -> 100% CASH (no position opened)
       -> If YES -> continue to normalization
    
    2. NORMALIZE: Apply allocation mode rules
       - cash_allocation: Deploy only among selected, rest CASH
       - full_deployment: Deploy 100% equally among selected
    """
    
    def __init__(self, purchase_threshold: float = 0.50, 
                 allocation_mode: str = "full_deployment"):
        self.purchase_threshold = purchase_threshold
        self.allocation_mode = allocation_mode
    
    def normalize(self, positions: np.ndarray, probabilities: np.ndarray, test_df: pd.DataFrame, 
                  allocation_mode: str = None,
                  purchase_threshold: float = None) -> np.ndarray:
        alloc_mode = allocation_mode if allocation_mode is not None else self.allocation_mode
        threshold = purchase_threshold if purchase_threshold is not None else self.purchase_threshold
        
        is_binary = np.all((positions == 0) | (positions == 1))
        
        if is_binary:
            return self._normalize_binary_positions(
                positions, probabilities, test_df, alloc_mode, threshold
            )
        
        return self._normalize_weighted_positions(
            positions, probabilities, test_df, alloc_mode, threshold
        )
    
    def _normalize_binary_positions(self, positions: np.ndarray, 
                                    probabilities: np.ndarray,
                                    test_df: pd.DataFrame,
                                    allocation_mode: str,
                                    purchase_threshold: float) -> np.ndarray:
        test_df_copy = test_df.copy().reset_index(drop=True)
        test_df_copy["_pos"] = positions
        test_df_copy["_prob"] = probabilities
        
        normalized_vals = np.zeros_like(positions, dtype=float)
        
        for date, group in test_df_copy.groupby("date"):
            idx = group.index
            pos = group["_pos"].values
            prob = group["_prob"].values
            
            active_count = np.sum(pos > 0)
            best_signal = np.max(prob) if len(prob) > 0 else 0
            total_assets = len(pos)
            
            if best_signal < purchase_threshold or active_count == 0:
                normalized_vals[idx] = 0.0
            else:
                if allocation_mode == "cash_allocation":
                    weight = 1.0 / total_assets
                else: 
                    weight = 1.0 / active_count
                normalized_vals[idx] = pos * weight
                
        return normalized_vals
    
    def _normalize_weighted_positions(self, positions: np.ndarray,
                                      probabilities: np.ndarray,
                                      test_df: pd.DataFrame,
                                      allocation_mode: str,
                                      purchase_threshold: float) -> np.ndarray:
        test_df_copy = test_df.copy().reset_index(drop=True)
        test_df_copy["_pos"] = positions
        test_df_copy["_prob"] = probabilities
        
        normalized_vals = np.zeros_like(positions, dtype=float)
        
        for date, group in test_df_copy.groupby("date"):
            idx = group.index
            pos = group["_pos"].values
            prob = group["_prob"].values
            
            total = np.sum(pos)
            best_signal = np.max(prob) if len(prob) > 0 else 0
            
            if best_signal < purchase_threshold or total == 0:
                normalized_vals[idx] = 0.0
            else:
                if allocation_mode == "cash_allocation":
                    normalized_vals[idx] = pos
                else: 
                    normalized_vals[idx] = pos / total if total > 0 else pos
                    
        return normalized_vals
