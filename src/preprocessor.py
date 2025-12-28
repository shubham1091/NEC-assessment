
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Any


class NECPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom preprocessor for NEC plant selection data.
    
    Performs:
    - Feature validation and extraction (DF*, PF* columns)
    - Missing value imputation
    - Feature scaling (StandardScaler)
    - Plant filtering (remove worst performers)
    
    This transformer works on the raw DataFrames and returns processed data.
    """
    
    def __init__(self, 
                 plant_filter_percentile: float = 75,
                 missing_value_strategy: str = 'mean',
                 random_seed: int = 7042025):
        """
        Initialize preprocessor.
        
        Args:
            plant_filter_percentile: Percentile threshold for filtering plants
            missing_value_strategy: Strategy for handling missing values ('mean', 'median', 'zero')
            random_seed: Random seed for reproducibility
        """
        self.plant_filter_percentile = plant_filter_percentile
        self.missing_value_strategy = missing_value_strategy
        self.random_seed = random_seed
        
        # Fitted attributes (set during fit())
        self.demand_features_ = None
        self.plant_features_ = None
        self.scaler_demand_ = None
        self.scaler_plant_ = None
        self.good_plants_ = None
        self.worst_plants_ = None
        self.threshold_cost_ = None
        self.is_fitted_ = False
    
    def _validate_features(self, df: pd.DataFrame, prefix: str) -> List[str]:
        """Extract features with given prefix that are numeric"""
        features = [col for col in df.columns 
                   if col.startswith(prefix) and pd.api.types.is_numeric_dtype(df[col])]
        return features
    
    def fit(self, data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], y=None):
        """
        Fit the preprocessor on training data.
        
        Args:
            data: Tuple of (demand_df, plants_df, costs_df)
            y: Ignored (for sklearn compatibility)
            
        Returns:
            self
        """
        demand_df, plants_df, costs_df = data
        
        # 1. Identify features
        self.demand_features_ = self._validate_features(demand_df, 'DF')
        self.plant_features_ = self._validate_features(plants_df, 'PF')
        
        # 2. Fit scalers on demand and plant features
        self.scaler_demand_ = StandardScaler()
        self.scaler_plant_ = StandardScaler()
        
        self.scaler_demand_.fit(demand_df[self.demand_features_].fillna(
            demand_df[self.demand_features_].mean() if self.missing_value_strategy == 'mean'
            else demand_df[self.demand_features_].median()
        ))
        
        self.scaler_plant_.fit(plants_df[self.plant_features_])
        
        # 3. Analyze plant performance and determine filter threshold
        plant_stats = costs_df.groupby('Plant ID')['Cost_USD_per_MWh'].agg(
            ['median', 'mean', 'count']
        ).sort_values('median')
        
        self.threshold_cost_ = plant_stats['median'].quantile(
            self.plant_filter_percentile / 100
        )
        self.worst_plants_ = plant_stats[plant_stats['median'] > self.threshold_cost_].index.tolist()
        self.good_plants_ = plant_stats[plant_stats['median'] <= self.threshold_cost_].index.tolist()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Transform the data using fitted parameters.
        
        Args:
            data: Tuple of (demand_df, plants_df, costs_df)
            
        Returns:
            Tuple of (transformed_demand_df, transformed_plants_df, filtered_costs_df)
        """
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        demand_df, plants_df, costs_df = data
        
        # Make copies to avoid modifying originals
        demand_df = demand_df.copy()
        plants_df = plants_df.copy()
        costs_df = costs_df.copy()
        
        # 1. Handle missing values in demand features
        if self.missing_value_strategy == 'mean':
            demand_df[self.demand_features_] = demand_df[self.demand_features_].fillna(
                demand_df[self.demand_features_].mean()
            )
        elif self.missing_value_strategy == 'median':
            demand_df[self.demand_features_] = demand_df[self.demand_features_].fillna(
                demand_df[self.demand_features_].median()
            )
        else:  # zero
            demand_df[self.demand_features_] = demand_df[self.demand_features_].fillna(0)
        
        # 2. Scale features
        demand_df[self.demand_features_] = self.scaler_demand_.transform(
            demand_df[self.demand_features_]
        )
        plants_df[self.plant_features_] = self.scaler_plant_.transform(
            plants_df[self.plant_features_]
        )
        
        # 3. Filter plants and costs
        plants_df = plants_df[plants_df['Plant ID'].isin(self.good_plants_)].reset_index(drop=True)
        costs_df = costs_df[
            (costs_df['Demand ID'].isin(demand_df['Demand ID'])) &
            (costs_df['Plant ID'].isin(self.good_plants_))
        ].reset_index(drop=True)
        
        # 4. Remove NaN costs
        costs_df = costs_df.dropna(subset=['Cost_USD_per_MWh']).reset_index(drop=True)
        
        return demand_df, plants_df, costs_df
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            'demand_features_count': len(self.demand_features_),
            'plant_features_count': len(self.plant_features_),
            'total_plants_before': len(self.good_plants_) + len(self.worst_plants_),
            'good_plants_count': len(self.good_plants_),
            'filtered_plants_count': len(self.worst_plants_),
            'cost_threshold': self.threshold_cost_,
            'plant_filter_percentile': self.plant_filter_percentile,
            'missing_value_strategy': self.missing_value_strategy
        }
