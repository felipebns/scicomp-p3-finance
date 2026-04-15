import pandas as pd
import numpy as np
from services.logger_config import get_logger

logger = get_logger()


class PositionNormalizer:
    """Normalizes position vectors to capital fractions.
    
    Pipeline:
    1. GATE: Check if best_signal meets purchase_threshold (confidence gate)
       → If NO → 100% CASH (no position opened)
       → If YES → continue to normalization
    
    2. NORMALIZE: Apply allocation mode rules
       - cash_allocation: Deploy only among selected, rest CASH
       - full_deployment: Deploy 100% equally among selected
    
    Supports two allocation modes:
    1. "cash_allocation": If N stocks selected, invest in N, rest stays CASH
    2. "full_deployment": If N stocks selected, deploy 100% equally across N
    """
    
    def __init__(self, purchase_threshold: float = 0.50, 
                 allocation_mode: str = "full_deployment"):
        """Initialize normalizer with default parameters.
        
        Args:
            purchase_threshold: Minimum confidence to open any position (0-1)
            allocation_mode: "cash_allocation" or "full_deployment"
        """
        self.purchase_threshold = purchase_threshold
        self.allocation_mode = allocation_mode
    
    def normalize(self, positions: np.ndarray, test_df: pd.DataFrame, 
                  allocation_mode: str = None,
                  purchase_threshold: float = None) -> np.ndarray:
        """Convert positions to normalized capital fractions.
        
        Args:
            positions: Position array (0/1 for binary, [0,1] for weighted)
            test_df: DataFrame with date column
            allocation_mode: "cash_allocation" or "full_deployment" (uses default if None)
            purchase_threshold: Minimum confidence to open any position (uses default if None)
        
        Returns:
            Normalized position array
        """
        alloc_mode = allocation_mode if allocation_mode is not None else self.allocation_mode
        threshold = purchase_threshold if purchase_threshold is not None else self.purchase_threshold
        
        is_binary = np.all((positions == 0) | (positions == 1))
        
        if is_binary:
            return self._normalize_binary_positions(
                positions, test_df, alloc_mode, threshold
            )
        
        # For weighted positions, apply allocation mode logic
        return self._normalize_weighted_positions(
            positions, test_df, alloc_mode, threshold
        )
    
    def _normalize_binary_positions(self, positions: np.ndarray, 
                                    test_df: pd.DataFrame,
                                    allocation_mode: str,
                                    purchase_threshold: float) -> np.ndarray:
        """Normalize binary positions.
        
        For binary positions: both allocation modes result in equal weighting
        because we divide by active_count either way. The allocation_mode 
        parameter only affects weighted positions (see _normalize_weighted_positions).
        """
        test_df_copy = test_df.copy()
        test_df_copy["_pos"] = positions
        
        normalized = test_df_copy.groupby("date")["_pos"].transform(
            lambda group: self._normalize_by_date_binary(group, purchase_threshold)
        )
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
        
        normalized = test_df_copy.groupby("date")["_pos"].transform(
            lambda group: self._normalize_by_date_weighted(
                group, allocation_mode, purchase_threshold
            )
        )
        return normalized.values
    
    def _normalize_by_date_binary(self, group: pd.Series, 
                                   purchase_threshold: float) -> pd.Series:
        """Apply gate and normalization logic for binary positions per date.
        
        Args:
            group: Position values for a single date
            purchase_threshold: Minimum confidence to open any position
        
        Returns:
            Normalized positions for the date
        """
        active_count = np.sum(group > 0)
        
        # GATE: Check if best signal meets purchase threshold
        best_signal = np.max(group)
        if best_signal < purchase_threshold:
            # No signal strong enough → 100% CASH
            return pd.Series(np.zeros_like(group), index=group.index)
        
        # If no active positions despite passing threshold, stay CASH
        if active_count == 0:
            return group
        
        # NORMALIZE: Divide equally among active positions
        return group / active_count
    
    def _normalize_by_date_weighted(self, group: pd.Series,
                                     allocation_mode: str,
                                     purchase_threshold: float) -> pd.Series:
        """Apply gate and normalization logic for weighted positions per date.
        
        Args:
            group: Position values for a single date
            allocation_mode: "cash_allocation" or "full_deployment"
            purchase_threshold: Minimum confidence to open any position
        
        Returns:
            Normalized positions for the date
        """
        total = np.sum(group)
        
        # GATE: Check if best signal meets purchase threshold
        best_signal = np.max(group)
        if best_signal < purchase_threshold:
            # No signal strong enough → 100% CASH
            return pd.Series(np.zeros_like(group), index=group.index)
        
        if total == 0:
            return group
        
        # NORMALIZE: Apply allocation mode
        if allocation_mode == "cash_allocation":
            # Keep proportional weights, don't force 100%
            return group
        else:  # full_deployment
            # Normalize to 100% if any position exists
            return group / total if total > 0 else group

