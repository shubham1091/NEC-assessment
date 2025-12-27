import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration container for NEC ML Pipeline"""
    
    # Data paths
    demand_path: str
    plants_path: str
    costs_path: str
    
    # Model selection
    model_type: str
    random_seed: int
    
    # Preprocessing
    plant_filter_percentile: float
    missing_value_strategy: str
    
    # Train/test split
    test_size_percent: float
    
    # Cross-validation
    cv_n_folds: int
    cv_n_jobs: int
    
    # Model hyperparameters
    model_params: Dict[str, Any]
    tuning_grid: Dict[str, Any]
    
    # Hyperparameter optimization
    hyperparam_enabled: bool
    hyperparam_cv_folds: int
    hyperparam_scoring: str
    
    # Output directories
    artifacts_dir: str
    results_dir: str
    logs_dir: str
    verbose: bool


def load_config(config_path: str = "config.yaml") -> Config:
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Validate model type
    valid_models = ['random_forest', 'gradient_boosting', 'lasso']
    model_type = yaml_config.get('model_type', 'random_forest')
    if model_type not in valid_models:
        raise ValueError(f"Invalid model_type: {model_type}. Must be one of {valid_models}")
    
    # Extract model-specific parameters
    model_config = yaml_config['models'][model_type]
    
    # Build Config object
    config = Config(
        # Data paths
        demand_path=yaml_config['data']['demand_path'],
        plants_path=yaml_config['data']['plants_path'],
        costs_path=yaml_config['data']['costs_path'],
        
        # Model selection
        model_type=model_type,
        random_seed=yaml_config['random_seed'],
        
        # Preprocessing
        plant_filter_percentile=yaml_config['preprocessing']['plant_filter_percentile'],
        missing_value_strategy=yaml_config['preprocessing']['missing_value_strategy'],
        
        # Train/test split
        test_size_percent=yaml_config['train_test']['test_size_percent'],
        
        # Cross-validation
        cv_n_folds=yaml_config['cross_validation']['n_folds'],
        cv_n_jobs=yaml_config['cross_validation']['n_jobs'],
        
        # Model hyperparameters
        model_params=model_config['default'],
        tuning_grid=model_config['tuning_grid'],
        
        # Hyperparameter optimization
        hyperparam_enabled=yaml_config['hyperparameter_optimization']['enabled'],
        hyperparam_cv_folds=yaml_config['hyperparameter_optimization']['cv_folds'],
        hyperparam_scoring=yaml_config['hyperparameter_optimization']['scoring'],
        
        # Output directories
        artifacts_dir=yaml_config['output']['artifacts_dir'],
        results_dir=yaml_config['output']['results_dir'],
        logs_dir=yaml_config['output']['logs_dir'],
        verbose=yaml_config['output']['verbose']
    )
    
    return config


def save_effective_config(config: Config, output_path: str):

    config_dict = {
        'data': {
            'demand_path': config.demand_path,
            'plants_path': config.plants_path,
            'costs_path': config.costs_path
        },
        'model_type': config.model_type,
        'random_seed': config.random_seed,
        'preprocessing': {
            'plant_filter_percentile': config.plant_filter_percentile,
            'missing_value_strategy': config.missing_value_strategy
        },
        'train_test': {
            'test_size_percent': config.test_size_percent
        },
        'cross_validation': {
            'n_folds': config.cv_n_folds,
            'n_jobs': config.cv_n_jobs
        },
        'model_params': config.model_params,
        'tuning_grid': config.tuning_grid,
        'hyperparameter_optimization': {
            'enabled': config.hyperparam_enabled,
            'cv_folds': config.hyperparam_cv_folds,
            'scoring': config.hyperparam_scoring
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
