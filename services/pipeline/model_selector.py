import time
import pandas as pd
import numpy as np
from typing import Tuple, Dict

from services.validation.walk_forward_validator import WalkForwardValidator
from services.algorithms.base import Algorithm


class ModelSelector:
    """Selects best ML model using walk-forward validation and Information Coefficient."""
    
    def __init__(self, wfv_train_window: int, wfv_test_window: int):
        self.validator = WalkForwardValidator(wfv_train_window, wfv_test_window)
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        self.results = {}
    
    def select(self, algorithms: list[Algorithm], train_df: pd.DataFrame, 
               features: list[str], target_col: str) -> Tuple[Algorithm, str, float]:
        """
        Select best algorithm using walk-forward validation.
        
        Returns:
            (best_algorithm, algorithm_name, best_score)
        """
        for algo in algorithms:
            print(f"\n[Phase 1: Selection] WFV for {algo.name()}...")
            start_time = time.time()
            
            try:
                wfv_df, wfv_preds, fold_ics, mean_ic, std_ic = self.validator.validate(
                    train_df, algo, features, target_col
                )
                
                # Robust score: mean IC - penalty for high variance
                score = mean_ic - (0.5 * std_ic)
                
            except Exception as e:
                print(f"  -> WFV failed: {e}. Score = 0.")
                score = 0.0
                mean_ic = 0.0
                std_ic = 0.0
                fold_ics = []
            
            elapsed = time.time() - start_time
            
            # Print results
            self._print_wfv_results(algo.name(), fold_ics, mean_ic, std_ic, score, elapsed)
            
            # Update if better
            if score > self.best_score:
                self.best_score = score
                self.best_model = algo
                self.best_model_name = algo.name()
            
            # Store metrics for reporting
            self.results[algo.name()] = {
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "fold_ics": fold_ics,
                "robust_score": score
            }
        
        return self.best_model, self.best_model_name, self.best_score
    
    def _print_wfv_results(self, name: str, fold_ics: list, mean_ic: float, 
                          std_ic: float, score: float, elapsed: float) -> None:
        """Print walk-forward validation results."""
        print(f"  -> Fold ICs:     {[round(ic, 4) for ic in fold_ics]}")
        print(f"  -> Mean IC:      {mean_ic:.6f} ± {std_ic:.6f}")
        print(f"  -> Robust Score: {score:.6f}")
        print(f"  -> Time:         {elapsed:.2f}s")
