import pandas as pd
import numpy as np
from services.log.logger_config import get_logger

logger = get_logger()


class AllocationManager:
    """
    Capital allocation orchestrator.
    
    Consolidates the three-step allocation pipeline while maintaining
    the exact logic of the original implementation:
    1. Top-K asset selection
    2. Probability weighting
    3. Position normalization (with confidence gate)
    
    Separates allocation logic from backtesting engine for better
    testability and maintainability.
    """
    
    def __init__(self, test_df: pd.DataFrame, 
                 position_selection: str = "top_5",
                 position_sizing: str = "equal_weight",
                 allocation_mode: str = "full_deployment",
                 purchase_threshold: float = 0.50):
        """
        Initialize allocation manager.
        
        Args:
            test_df: DataFrame with date and ticker columns
            position_selection: "all", "top_1", "top_5", etc.
            position_sizing: "equal_weight" or "probability_weighted"
            allocation_mode: "full_deployment" or "cash_allocation"
            purchase_threshold: Minimum confidence to invest (confidence gate)
        """
        self.test_df = test_df
        self.position_selection = position_selection
        self.position_sizing = position_sizing
        self.allocation_mode = allocation_mode
        self.purchase_threshold = purchase_threshold
    
    def allocate(self, positions: np.ndarray, probabilities: np.ndarray,
                 strategy_name: str) -> np.ndarray:
        """
        Execute complete allocation pipeline.
        
        Pipeline (same logic as original):
        1. Apply top-K filtering (skip for benchmarks)
        2. Apply probability weighting if enabled
        3. Normalize positions (apply allocation mode + confidence gate)
        
        Args:
            positions: Binary positions (0/1) from strategy
            probabilities: Model probabilities for weighting
            strategy_name: Strategy name (skip allocation for benchmarks)
        
        Returns:
            Allocated and normalized positions
        """
        n_selected_before = np.sum(positions > 0)
        
        # STEP 1: Top-K filtering (skip for benchmarks)
        if strategy_name not in ["fixed_income", "buy_and_hold"]:
            positions = self._apply_position_selection(positions, probabilities)
            n_selected_after_topk = np.sum(positions > 0)
            logger.debug(f"[{strategy_name}] Top-K Filter ({self.position_selection}): "
                        f"{n_selected_before} → {n_selected_after_topk} selected")
        
        # STEP 2: Probability weighting (skip for benchmarks)
        if self.position_sizing == "probability_weighted" and \
            strategy_name not in ["fixed_income", "buy_and_hold", "simple_reversal"]:
            # Weight the binary positions by their probabilities
            positions = positions * probabilities
            positions = self._apply_probability_weights(positions)
            logger.debug(f"[{strategy_name}] Probability weighting applied (weighted mode)")
        
        # STEP 3: Normalize positions (apply allocation mode + confidence gate)
        positions = self._normalize_positions(positions)
        n_final = np.sum(positions > 0)
        logger.debug(f"[{strategy_name}] Normalization complete: "
                    f"allocation_mode={self.allocation_mode}, "
                    f"purchase_threshold={self.purchase_threshold}, "
                    f"final_active={n_final}")
        
        return positions
    
    def _apply_position_selection(self, positions: np.ndarray, 
                                   probabilities: np.ndarray) -> np.ndarray:
        """Filter positions keeping only top-K by probability.
        
        This is EXACT copy of the original backtest._apply_position_selection
        
        Args:
            positions: Position array (0 or 1 for each row)
            probabilities: Model probability for each row
        
        Returns:
            Filtered position array
        """
        if self.position_selection == "all":
            return positions
        
        # Parse "top_5" → 5
        if not self.position_selection.startswith("top_"):
            return positions
        
        try:
            n_top = int(self.position_selection.split("_")[1])
        except (IndexError, ValueError):
            return positions
        
        filtered_positions = positions.copy()
        dates = self.test_df["date"].unique()
        
        for date in dates:
            mask = self.test_df["date"] == date
            date_positions = positions[mask.values]
            date_probs = probabilities[mask.values]
            
            # Only apply filter if there are positions to take
            n_positions = np.sum(date_positions > 0)
            
            if n_positions > n_top:
                # Get indices of top N positions by probability (local to this date)
                position_indices = np.where(date_positions > 0)[0]
                position_probs = date_probs[position_indices]
                
                # Sort by probability descending, take top N (still local indices)
                local_top_indices = position_indices[np.argsort(-position_probs)[:n_top]]
                
                # Zero out all positions for this date
                filtered_positions[mask.values] = 0
                
                # Re-add only the top N (using local indices)
                global_indices = np.where(mask.values)[0][local_top_indices]
                filtered_positions[global_indices] = 1
        
        return filtered_positions
    
    def _apply_probability_weights(self, probabilities: np.ndarray) -> np.ndarray:
        """Normalize probabilities per date.
        
        This is EXACT copy of the original backtest._apply_probability_weights
        
        Args:
            probabilities: Array of probability values
        
        Returns:
            Normalized probability weights per date
        """
        weights = np.zeros_like(probabilities)
        dates = self.test_df["date"].unique()
        
        for date in dates:
            mask = self.test_df["date"] == date
            date_probs = probabilities[mask]
            total_prob = np.sum(date_probs[date_probs > 0])
            
            if total_prob > 0:
                weights[mask] = np.where(
                    date_probs > 0,
                    date_probs / total_prob,
                    0
                )
        
        return weights
    
    def _normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        """Normalize positions based on allocation mode.
        
        This delegates to PositionNormalizer to maintain exact original logic.
        The confidence gate is applied INSIDE the normalization, which is the
        critical difference from the broken refactored version.
        
        Args:
            positions: Position array (binary or weighted)
        
        Returns:
            Normalized positions respecting allocation mode and confidence gate
        """
        from services.backtesting.position_normalizer import PositionNormalizer
        
        normalizer = PositionNormalizer()
        return normalizer.normalize(
            positions, 
            self.test_df,
            allocation_mode=self.allocation_mode,
            purchase_threshold=self.purchase_threshold
        )
