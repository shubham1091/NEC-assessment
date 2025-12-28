import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from typing import Dict, Any, Tuple, List
import logging

from .scorer import calculate_plant_selection_error_grouped, calculate_error_statistics, calculate_rmse


logger = logging.getLogger(__name__)


class Evaluator:
    """Handles model evaluation with grouped train/test and LOGO CV"""
    
    def __init__(self, random_seed: int = 7042025):
        """
        Initialize evaluator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
    
    def create_combined_dataset(self,
                                demand_df: pd.DataFrame,
                                plants_df: pd.DataFrame,
                                costs_df: pd.DataFrame,
                                demand_features: List[str],
                                plant_features: List[str]) -> pd.DataFrame:
        """
        Create combined dataset with demand+plant features and costs.
        
        Args:
            demand_df: Demand data (scaled)
            plants_df: Plant data (scaled)
            costs_df: Cost data (filtered)
            demand_features: List of demand feature column names
            plant_features: List of plant feature column names
            
        Returns:
            Combined DataFrame with all features and cost
        """
        logger.info("Creating combined dataset...")
        
        combined_data = []
        for _, cost_row in costs_df.iterrows():
            demand_id = cost_row["Demand ID"]
            plant_id = cost_row["Plant ID"]
            cost = cost_row["Cost_USD_per_MWh"]
            
            demand_row = demand_df[demand_df["Demand ID"] == demand_id].iloc[0]
            plant_row = plants_df[plants_df["Plant ID"] == plant_id].iloc[0]
            
            row = {"Demand_ID": demand_id, "Plant_ID": plant_id}
            for feat in demand_features:
                row[feat] = demand_row[feat]
            for feat in plant_features:
                row[feat] = plant_row[feat]
            row['Cost'] = cost
            
            combined_data.append(row)
        
        combined_df = pd.DataFrame(combined_data)
        
        # Drop any NaN rows
        initial_len = len(combined_df)
        combined_df = combined_df.dropna().reset_index(drop=True)
        dropped = initial_len - len(combined_df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with NaN values")
        
        logger.info(f"Combined dataset created: {len(combined_df)} records, "
                   f"{combined_df['Demand_ID'].nunique()} demands, "
                   f"{combined_df['Plant_ID'].nunique()} plants")
        
        return combined_df
    
    def grouped_train_test_split(self,
                                 combined_df: pd.DataFrame,
                                 test_size_percent: float,
                                 demand_features: List[str],
                                 plant_features: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, set, pd.DataFrame]:
        """
        Perform grouped train/test split by Demand ID.
        
        Args:
            combined_df: Combined dataset
            test_size_percent: Percentage of demands for test set
            demand_features: List of demand features
            plant_features: List of plant features
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, test_demand_ids, combined_df_test)
        """
        logger.info(f"Performing grouped train/test split (test size: {test_size_percent}%)...")
        
        all_features = demand_features + plant_features
        
        # Use GroupShuffleSplit to ensure demand IDs are not split between train and test
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size_percent / 100, random_state=self.random_seed)
        train_idx, test_idx = next(gss.split(combined_df, groups=combined_df["Demand_ID"]))
        
        test_demand_ids = set(combined_df.iloc[test_idx]["Demand_ID"].unique())
        
        X_train = combined_df.iloc[train_idx][all_features].values.astype(np.float64)
        y_train = combined_df.iloc[train_idx]["Cost"].values.astype(np.float64)
        X_test = combined_df.iloc[test_idx][all_features].values.astype(np.float64)
        y_test = combined_df.iloc[test_idx]["Cost"].values.astype(np.float64)
        
        combined_df_test = combined_df.iloc[test_idx].copy()
        
        logger.info(f"Train set: {len(X_train)} samples, {combined_df.iloc[train_idx]['Demand_ID'].nunique()} demands")
        logger.info(f"Test set: {len(X_test)} samples, {len(test_demand_ids)} demands")
        
        return X_train, X_test, y_train, y_test, test_demand_ids, combined_df_test
    
    def evaluate_on_test_set(self,
                            model,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            combined_df_test: pd.DataFrame,
                            demand_features: List[str],
                            plant_features: List[str]) -> Tuple[np.ndarray, Dict[str, Any], pd.DataFrame]:
        """
        Evaluate model on held-out test set and generate per-scenario selection table.
        
        Args:
            model: Fitted model
            X_test: Test features
            y_test: Test costs
            combined_df_test: Test set DataFrame
            demand_features: List of demand features
            plant_features: List of plant features
            
        Returns:
            Tuple of (errors, error_stats, scenario_table)
        """
        logger.info("Evaluating on test set...")
        
        y_pred = model.predict(X_test)
        demand_ids = combined_df_test['Demand_ID'].to_numpy(copy=False)
        
        # Calculate selection errors
        errors = calculate_plant_selection_error_grouped(y_test, y_pred, demand_ids)
        error_stats = calculate_error_statistics(errors)
        
        logger.info(f"Test set RMSE: ${error_stats['rmse']:.2f}/MWh")
        logger.info(f"Test set mean error: ${error_stats['mean']:.2f}/MWh")
        
        # Generate per-scenario selection table
        scenario_table = self._generate_scenario_table(
            y_test, y_pred, combined_df_test
        )
        
        return errors, error_stats, scenario_table
    
    def _generate_scenario_table(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 combined_df_test: pd.DataFrame) -> pd.DataFrame:
        """Generate per-scenario selection table"""
        df = combined_df_test.copy()
        df['Predicted_Cost'] = pd.to_numeric(y_pred, errors='coerce')
        df['Actual_Cost'] = pd.to_numeric(y_true, errors='coerce')
        
        scenario_records = []
        
        for demand_id, group in df.groupby('Demand_ID'):
            # Oracle selection
            oracle_idx = group['Actual_Cost'].idxmin()
            oracle_plant = group.loc[oracle_idx, 'Plant_ID']
            oracle_cost = float(pd.to_numeric(group.loc[oracle_idx, 'Actual_Cost'], errors='coerce'))
            
            # Model selection
            model_idx = group['Predicted_Cost'].idxmin()
            model_plant = group.loc[model_idx, 'Plant_ID']
            model_cost = float(pd.to_numeric(group.loc[model_idx, 'Actual_Cost'], errors='coerce'))
            
            # Error
            error = model_cost - oracle_cost
            
            scenario_records.append({
                'Demand_ID': demand_id,
                'Model_Selected_Plant': model_plant,
                'Oracle_Selected_Plant': oracle_plant,
                'Model_Selected_Cost': model_cost,
                'Oracle_Cost': oracle_cost,
                'Selection_Error': error
            })
        
        return pd.DataFrame(scenario_records)
    
    def logo_cross_validation(self,
                             model_class,
                             model_params: Dict[str, Any],
                             combined_df: pd.DataFrame,
                             demand_features: List[str],
                             plant_features: List[str],
                             n_folds: int = 500,
                             n_jobs: int = -1) -> Tuple[List[float], pd.DataFrame]:
        """
        Perform Leave-One-Group-Out cross-validation.
        
        Each fold leaves out one demand (all its plant combinations) for testing.
        
        Args:
            model_class: Model class (not instance)
            model_params: Model hyperparameters
            combined_df: Combined dataset
            demand_features: List of demand features
            plant_features: List of plant features
            n_folds: Number of folds to use (max = number of unique demands)
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuple of (fold_errors, fold_results_df)
        """
        logger.info(f"Performing LOGO cross-validation ({n_folds} folds)...")
        
        all_features = demand_features + plant_features
        X = combined_df[all_features].to_numpy(dtype=np.float64)
        y = combined_df['Cost'].to_numpy(dtype=np.float64)
        # Extract numeric part from Demand_ID (e.g., 'D1' -> 1)
        groups = np.asarray(combined_df['Demand_ID'].str.extract('(\d+)')[0].astype(np.int64).values, dtype=np.int64)
        
        logo = LeaveOneGroupOut()
        all_splits = list(logo.split(X, y, groups))
        
        # Use only first n_folds
        splits_to_use = all_splits[:min(n_folds, len(all_splits))]
        
        logger.info(f"Running {len(splits_to_use)} LOGO CV folds...")
        
        fold_errors = []
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits_to_use):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_test = groups[test_idx]
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate error for this fold (one demand)
            errors = calculate_plant_selection_error_grouped(y_test, y_pred, groups_test)
            fold_error = errors[0] if len(errors) > 0 else 0.0
            fold_rmse = calculate_rmse(errors) if len(errors) > 0 else 0.0
            
            fold_errors.append(fold_error)
            
            fold_results.append({
                'Fold': fold_idx + 1,
                'Demand_ID': groups_test[0],
                'Selection_Error': fold_error,
                'Fold_RMSE': fold_rmse
            })
            
            if (fold_idx + 1) % 50 == 0:
                logger.info(f"Completed {fold_idx + 1}/{len(splits_to_use)} folds")
        
        fold_results_df = pd.DataFrame(fold_results)
        
        overall_rmse = calculate_rmse(np.array(fold_errors))
        logger.info(f"LOGO CV completed. Overall RMSE: ${overall_rmse:.2f}/MWh")
        
        return fold_errors, fold_results_df
