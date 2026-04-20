import os
import random
import numpy as np


class ReproducibilityManager:
    """Centralizes all randomness control for deterministic results."""
    
    @classmethod
    def setup_reproducibility(cls, seed: int = 42) -> None:
        """
        Setup reproducibility by setting all random seeds.
        
        CRITICAL: Must be called BEFORE importing ML libraries.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)