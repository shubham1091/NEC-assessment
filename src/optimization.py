"""
Hyperparameter optimization module for NEC ML Pipeline.
Implements grid search with LOGO CV and custom scorer.

Team Contribution: Shubham - Hyperparameter optimization and tuning framework
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
import logging
from typing import Dict, Any, List, Tuple
from itertools import product

from .scorer import calculate_plant_selection_error_grouped, calculate_rmse


logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Handles hyperparameter optimization using LOGO CV with custom scorer"""
    
    def __init__(self, random_seed: int = 7042025):
        """
        Initialize optimizer.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.best_params_ = None
        self.best_score_ = None
        self.leaderboard_ = None
    
    def optimize(self,
                 model_class,
                 param_grid: Dict[str, List],
                 combined_df: pd.DataFrame,
                 demand_features: List[str],
                 plant_features: List[str],
                 n_cv_folds: int = 50) -> Tuple[Dict[str, Any], float, pd.DataFrame]:
        """
        Perform grid search with LOGO cross-validation.
        
        Args:
            model_class: Model class (not instance)
            param_grid: Dictionary of parameter names and values to search
            combined_df: Combined dataset
            demand_features: List of demand features
            plant_features: List of plant features
            n_cv_folds: Number of CV folds to use
            
        Returns:
            Tuple of (best_params, best_rmse, leaderboard_df)
        """
        logger.info("Starting hyperparameter optimization...")
        logger.info(f"Parameter grid: {param_grid}")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))
        
        logger.info(f"Total combinations to evaluate: {len(all_combinations)}")
        
        # Prepare data
        all_features = demand_features + plant_features
        X = combined_df[all_features].values.astype(np.float64)
        y = combined_df['Cost'].values.astype(np.float64)
        # Extract numeric part from Demand_ID (e.g., 'D1' -> 1)
        groups = np.asarray(combined_df['Demand_ID'].str.extract('(\d+)')[0].astype(np.int64).values, dtype=np.int64)
        
        # Setup LOGO CV
        logo = LeaveOneGroupOut()
        all_splits = list(logo.split(X, None, groups))
        splits_to_use = all_splits[:n_cv_folds]
        
        logger.info(f"Using {len(splits_to_use)} CV folds for optimization")
        
        # Evaluate each parameter combination
        results = []
        best_rmse = float('inf')
        best_params: Dict[str, Any] = {}
        
        for combo_idx, combo in enumerate(all_combinations, 1):
            # Build parameter dictionary
            params = dict(zip(param_names, combo))
            params['random_state'] = self.random_seed
            
            # Run LOGO CV with these parameters
            fold_errors = []
            
            for train_idx, test_idx in splits_to_use:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                groups_test = groups[test_idx]
                
                # Train model
                model = model_class(**params)
                model.fit(X_train, y_train)
                
                # Predict and calculate error
                y_pred = model.predict(X_test)
                errors = calculate_plant_selection_error_grouped(y_test, y_pred, groups_test)
                # Calculate RMSE from all demand errors (not just first one)
                fold_rmse = calculate_rmse(errors)
                fold_errors.append(fold_rmse)
            
            # Calculate mean RMSE across folds
            fold_errors_array = np.array(fold_errors)
            mean_rmse = calculate_rmse(fold_errors_array)
            std_rmse = np.std(fold_errors_array)
            
            # Record results
            result = {
                'params': str(params),
                'mean_rmse': mean_rmse,
                'std_rmse': std_rmse
            }
            for param_name, param_value in params.items():
                if param_name != 'random_state':
                    result[param_name] = param_value
            
            results.append(result)
            
            # Update best
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_params = {k: v for k, v in params.items() if k != 'random_state'}
            
            logger.info(f"[{combo_idx}/{len(all_combinations)}] {params} -> RMSE: ${mean_rmse:.2f} ± ${std_rmse:.2f}")
        
        # Create leaderboard
        leaderboard_df = pd.DataFrame(results)
        leaderboard_df = leaderboard_df.sort_values('mean_rmse').reset_index(drop=True)
        leaderboard_df['rank'] = range(1, len(leaderboard_df) + 1)
        
        self.best_params_ = best_params
        self.best_score_ = best_rmse
        self.leaderboard_ = leaderboard_df
        
        logger.info(f"Optimization complete!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best RMSE: ${best_rmse:.2f}/MWh")
        
        return best_params, best_rmse, leaderboard_df
