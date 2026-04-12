import pandas as pd
import numpy as np


class PositionNormalizer:
    """Normalizes position vectors to capital fractions."""
    
    def normalize(self, positions: np.ndarray, test_df: pd.DataFrame) -> np.ndarray:
        """
        Convert positions to normalized capital fractions.
        
        - Binary (0/1): Normalize per date so active assets get equal weight
        - Weighted [0,1]: Already normalized, return as-is
        """
        is_binary = np.all((positions == 0) | (positions == 1))
        
        if is_binary:
            return self._normalize_binary_positions(positions, test_df)
        return positions
    
    def _normalize_binary_positions(self, positions: np.ndarray, 
                                    test_df: pd.DataFrame) -> np.ndarray:
        """For binary positions, divide equally among active assets per date."""
        test_df_copy = test_df.copy()
        test_df_copy["_pos"] = positions
        
        def normalize_by_date(group):
            active_count = np.sum(group > 0)
            return group / active_count if active_count > 0 else group
        
        normalized = test_df_copy.groupby("date")["_pos"].transform(normalize_by_date)
        return normalized.values
