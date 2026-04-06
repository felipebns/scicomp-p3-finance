import json
import itertools
from pathlib import Path
from typing import Dict, List, Any

from services.pipeline import Pipeline

class HyperparameterSweep:
    """Hyperparameter tuning using reusable Pipeline objects."""
    
    def __init__(self, algorithm_class, stock, param_grid: Dict[str, List[Any]], 
                 output_path: str = "output/hyperparameter_sweep.json",
                 metric_to_optimize: str = "r2"):
        self.algorithm_class = algorithm_class
        self.stock = stock
        self.param_grid = param_grid
        self.output_path = output_path
        self.metric_to_optimize = metric_to_optimize
        self.results = []
        self.best_config = None
        self.best_metric_value = float('-inf') if metric_to_optimize == "r2" else float('inf')
    
    def _generate_combinations(self) -> List[Dict[str, Any]]:
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)
        return combinations
    
    def _evaluate_config(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Run the full pipeline and compute multiple metrics."""
        try:
            algorithm = self.algorithm_class(**config)
            pipeline = Pipeline(stock=self.stock, algorithm=algorithm, output_dir="output")
            
            metrics = pipeline.run(save=False)
            
            return {
                "rmse": float(metrics["test"]["rmse"]),
                "mae": float(metrics["test"]["mae"]),
                "r2": float(metrics["test"]["r2"])
            }
            
        except Exception as e:
            print(f"  ⚠️  Error with config {config}: {str(e)}")
            return {"rmse": float('inf'), "mae": float('inf'), "r2": float('-inf')}
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Run grid sweep."""
        combinations = self._generate_combinations()
        total = len(combinations)
        
        if verbose:
            print(f"\n🔍 Starting hyperparameter sweep ({total} combinations)...")
            print(f"   Optimizing for: {self.metric_to_optimize.upper()}\n")
        
        for idx, config in enumerate(combinations, 1):
            if verbose:
                config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
                print(f"[{idx}/{total}] Testing: {config_str}")
            
            metrics = self._evaluate_config(config)
            
            result = {
                "config": config,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"]
            }
            self.results.append(result)
            
            if verbose:
                print(f"  ✓ RMSE: {metrics['rmse']:.6f} | MAE: {metrics['mae']:.6f} | R²: {metrics['r2']:.6f}\n")
            
            # Track best based on chosen metric
            current_value = metrics[self.metric_to_optimize]
            is_better = False
            
            if self.metric_to_optimize == "r2":
                is_better = current_value > self.best_metric_value
            else:  # rmse, mae
                is_better = current_value < self.best_metric_value
            
            if is_better:
                self.best_metric_value = current_value
                self.best_config = config
        
        return self._get_results_summary()
    
    def _get_results_summary(self) -> Dict[str, Any]:
        sorted_results = sorted(
            self.results,
            key=lambda x: x[self.metric_to_optimize],
            reverse=(self.metric_to_optimize == "r2")
        )
        
        return {
            "best_config": self.best_config,
            f"best_{self.metric_to_optimize}": float(self.best_metric_value),
            "total_combinations_tested": len(self.results),
            "metric_optimized": self.metric_to_optimize,
            "top_5_results": sorted_results[:5],
            "all_results": self.results
        }
    
    def save_results(self):
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        summary = self._get_results_summary()
        with open(self.output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Results saved to {self.output_path}")
        print(f"Best Config: {summary['best_config']}")
        print(f"Best {self.metric_to_optimize.upper()}: {summary[f'best_{self.metric_to_optimize}']:.6f}")
