import pandas as pd
import numpy as np


class AllocationManager:
    """
    Capital allocation orchestrator.
    
    Centralized responsibility for:
    1. Top-K asset selection (position_selection)
    2. Probability weighting (position_sizing)
    3. Position normalization (allocation_mode)
    
    Separates allocation logic from backtesting engine.
    """
    
    def __init__(self, test_df: pd.DataFrame, 
                 position_selection: str = "top_5",
                 position_sizing: str = "equal_weight",
                 allocation_mode: str = "full_deployment",
                 min_assets_for_investment: int = 1,
                 portfolio_confidence_threshold: float = 0.50):
        """
        Initialize allocation manager.
        
        Args:
            test_df: DataFrame with date and ticker columns
            position_selection: "all", "top_1", "top_5", etc.
            position_sizing: "equal_weight" or "probability_weighted"
            allocation_mode: "full_deployment" or "cash_allocation"
            min_assets_for_investment: Minimum selected assets to deploy capital
            portfolio_confidence_threshold: Minimum confidence to invest (global gate)
        """
        self.test_df = test_df
        self.position_selection = position_selection
        self.position_sizing = position_sizing
        self.allocation_mode = allocation_mode
        self.min_assets_for_investment = min_assets_for_investment
        self.portfolio_confidence_threshold = portfolio_confidence_threshold
    
    def allocate(self, positions: np.ndarray, probabilities: np.ndarray,
                 strategy_name: str) -> np.ndarray:
        """
        Execute complete allocation pipeline.
        
        Pipeline:
        1. Apply top-K filtering
        2. Apply probability weighting
        3. Apply portfolio confidence gate (global minimum)
        4. Normalize positions
        
        Args:
            positions: Binary positions (0/1) from strategy
            probabilities: Model probabilities for weighting
            strategy_name: Strategy name (skip allocation for benchmarks)
        
        Returns:
            Allocated and normalized positions
        """
        # STEP 1: Top-K filtering (skip for benchmarks)
        if strategy_name not in ["fixed_income", "buy_and_hold"]:
            positions = self._apply_top_k_filter(positions, probabilities)
        
        # STEP 2: Probability weighting (skip for benchmarks)
        if self.position_sizing == "probability_weighted" and \
            strategy_name not in ["fixed_income", "buy_and_hold", "simple_reversal"]:
            positions = positions * probabilities
            positions = self._normalize_probability_weights(positions)
        
        # STEP 3: Apply portfolio confidence gate (skip for benchmarks)
        # If max probability for a date is below threshold → 100% CASH
        if strategy_name not in ["fixed_income", "buy_and_hold"]:
            positions = self._apply_confidence_gate(positions, probabilities)
        
        # STEP 4: Normalize positions (apply allocation mode)
        positions = self._normalize_positions(positions)
        
        return positions
    
    def _apply_top_k_filter(self, positions: np.ndarray, 
                           probabilities: np.ndarray) -> np.ndarray:
        """
        Filter positions keeping only top-K by probability.
        
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
            
            n_positions = np.sum(date_positions > 0)
            
            if n_positions > n_top:
                # Keep only top N by probability
                position_indices = np.where(date_positions > 0)[0]
                position_probs = date_probs[position_indices]
                
                local_top_indices = position_indices[np.argsort(-position_probs)[:n_top]]
                
                filtered_positions[mask.values] = 0
                global_indices = np.where(mask.values)[0][local_top_indices]
                filtered_positions[global_indices] = 1
        
        return filtered_positions
    
    def _apply_confidence_gate(self, positions: np.ndarray, 
                               probabilities: np.ndarray) -> np.ndarray:
        """
        Apply global confidence gate: if max probability for a date < threshold, go 100% CASH.
        
        Args:
            positions: Position array (0 or 1 for each row, or weighted)
            probabilities: Model probability for each row
        
        Returns:
            Positions with confidence gate applied (0 if below threshold for that date)
        """
        gated_positions = positions.copy()
        dates = self.test_df["date"].unique()
        
        for date in dates:
            mask = self.test_df["date"] == date
            date_probs = probabilities[mask.values]
            
            # Get max probability for this date
            max_prob = np.max(date_probs) if len(date_probs) > 0 else 0.0
            
            # If max probability below threshold → liquidate all positions for this date
            if max_prob < self.portfolio_confidence_threshold:
                gated_positions[mask.values] = 0
        
        return gated_positions
    
    def _normalize_probability_weights(self, probabilities: np.ndarray) -> np.ndarray:
        """Normalize probabilities per date to sum to 1."""
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
        """
        Normalize positions based on allocation mode.
        
        Architecture:
        1. Count selected assets per date
        2. Check minimum assets threshold
        3. Apply allocation mode (full_deployment vs cash_allocation)
        """
        test_df_copy = self.test_df.copy()
        test_df_copy["_pos"] = positions
        
        is_binary = np.all((positions == 0) | (positions == 1))
        
        if is_binary:
            return self._normalize_binary_positions(positions, test_df_copy)
        else:
            return self._normalize_weighted_positions(positions, test_df_copy)
    
    def _normalize_binary_positions(self, positions: np.ndarray,
                                   test_df_copy: pd.DataFrame) -> np.ndarray:
        """Normalize binary positions (0 or 1)."""
        def normalize_by_date(group):
            active_count = np.sum(group > 0)
            
            # No assets selected → 100% CASH
            if active_count == 0:
                return np.zeros_like(group)
            
            # Check minimum assets threshold
            if active_count < self.min_assets_for_investment:
                return np.zeros_like(group)
            
            # Allocate among selected
            if self.allocation_mode == "full_deployment":
                # Deploy 100% equally among selected
                return group / active_count
            elif self.allocation_mode == "cash_allocation":
                # Deploy proportionally, rest stays CASH
                return group / active_count
            else:
                return group / active_count
        
        normalized = test_df_copy.groupby("date")["_pos"].transform(normalize_by_date)
        return normalized.values
    
    def _normalize_weighted_positions(self, positions: np.ndarray,
                                     test_df_copy: pd.DataFrame) -> np.ndarray:
        """Normalize weighted positions (from probability multiplication)."""
        def normalize_by_date(group):
            total = np.sum(group)
            
            if total == 0:
                return np.zeros_like(group)
            
            active_count = np.sum(group > 0)
            
            if active_count < self.min_assets_for_investment:
                return np.zeros_like(group)
            
            if self.allocation_mode == "full_deployment":
                # Deploy 100% (normalize to sum = 1)
                return group / total
            elif self.allocation_mode == "cash_allocation":
                # Keep proportional weights (sum <= 1, rest is CASH)
                return group
            else:
                return group / total if total > 0 else group
        
        normalized = test_df_copy.groupby("date")["_pos"].transform(normalize_by_date)
        return normalized.values
