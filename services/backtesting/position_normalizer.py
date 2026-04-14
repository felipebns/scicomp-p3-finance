import pandas as pd
import numpy as np


class PositionNormalizer:
    """Normalizes position vectors to capital fractions.
    
    Supports two allocation modes:
    1. "cash_allocation": If N stocks selected, invest in N, rest stays CASH
    2. "full_deployment": If N stocks selected, deploy 100% equally across N
    """
    
    def normalize(self, positions: np.ndarray, test_df: pd.DataFrame, 
                  allocation_mode: str = "full_deployment",
                  purchase_threshold: float = 0.50) -> np.ndarray:
        """
        Convert positions to normalized capital fractions.
        
        Args:
            positions: Position array (0/1 for binary, [0,1] for weighted)
            test_df: DataFrame with date column
            allocation_mode: "cash_allocation" or "full_deployment"
            purchase_threshold: Minimum confidence to open any position
        
        Returns:
            Normalized position array
        """
        is_binary = np.all((positions == 0) | (positions == 1))
        
        if is_binary:
            return self._normalize_binary_positions(
                positions, test_df, allocation_mode, purchase_threshold
            )
        
        # For weighted positions, apply allocation mode logic
        return self._normalize_weighted_positions(
            positions, test_df, allocation_mode, purchase_threshold
        )
    
    def _normalize_binary_positions(self, positions: np.ndarray, 
                                    test_df: pd.DataFrame,
                                    allocation_mode: str,
                                    purchase_threshold: float) -> np.ndarray:
        """Normalize binary positions based on allocation mode.
        
        - cash_allocation: N stocks get 1/N each, rest is CASH
        - full_deployment: N stocks get 1/N each (100% deployed)
        """
        test_df_copy = test_df.copy()
        test_df_copy["_pos"] = positions
        
        def normalize_by_date(group):
            active_count = np.sum(group > 0)
            
            # Check if best signal meets purchase threshold
            best_signal = np.max(group)
            if best_signal < purchase_threshold:
                # No signal strong enough → 100% CASH
                return np.zeros_like(group)
            
            # If no active positions despite passing threshold, stay CASH
            if active_count == 0:
                return group
            
            # Divide equally among active positions
            if allocation_mode == "cash_allocation":
                # Invest only in selected, rest stays CASH
                return group / active_count
            else:  # full_deployment
                # Deploy 100% equally across selected
                return group / active_count
        
        normalized = test_df_copy.groupby("date")["_pos"].transform(normalize_by_date)
        return normalized.values
    
    def _normalize_weighted_positions(self, positions: np.ndarray,
                                      test_df: pd.DataFrame,
                                      allocation_mode: str,
                                      purchase_threshold: float) -> np.ndarray:
        """Normalize weighted positions based on allocation mode.
        
        For weighted positions (result of probability multiplication):
        - Verify they're normalized per date
        - Apply allocation mode if needed
        """
        test_df_copy = test_df.copy()
        test_df_copy["_pos"] = positions
        
        def normalize_by_date(group):
            total = np.sum(group)
            
            # Check if best signal meets purchase threshold
            best_signal = np.max(group)
            if best_signal < purchase_threshold:
                # No signal strong enough → 100% CASH
                return np.zeros_like(group)
            
            if total == 0:
                return group
            
            if allocation_mode == "cash_allocation":
                # Keep proportional weights, don't force 100%
                return group
            else:  # full_deployment
                # Normalize to 100% if any position exists
                return group / total if total > 0 else group
        
        normalized = test_df_copy.groupby("date")["_pos"].transform(normalize_by_date)
        return normalized.values

