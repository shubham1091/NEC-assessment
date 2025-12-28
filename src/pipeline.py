"""
Main NEC ML Pipeline orchestrator.
Coordinates all pipeline stages from data ingestion to artifact generation.

Team Contribution: All team members - Integrated pipeline design
"""

import pandas as pd
import numpy as np
from typing import cast

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .config import Config, load_config, save_effective_config
from .preprocessor import NECPreprocessor
from .evaluation import Evaluator
from .visualizations import Visualizer
from .models import ModelFactory
from .scorer import calculate_error_statistics
from .optimization import HyperparameterOptimizer


# Setup logging
def setup_logging(logs_dir: str, verbose: bool = True):
    """Setup logging configuration"""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(logs_dir) / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class NECPipeline:
    """Main pipeline orchestrator for NEC plant selection"""
    
    def __init__(self, config: Config):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        for dir_path in [config.artifacts_dir, config.results_dir, config.logs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Pipeline components
        self.preprocessor = None
        self.model = None
        self.evaluator = Evaluator(random_seed=config.random_seed)
        self.optimizer = HyperparameterOptimizer(random_seed=config.random_seed)
        
        # Results storage
        self.demand_df = None
        self.plants_df = None
        self.costs_df = None
        self.combined_df = None
        self.baseline_results = {}
        self.optimized_results = {}
        self.final_results = {}
    
    def run(self):
        """Execute the complete pipeline"""
        
        # Stage 1: Data Ingestion
        self._load_data()
        
        # Stage 2: Preprocessing
        self._preprocess_data()
        
        # Stage 3: Create Combined Dataset
        self._create_combined_dataset()
        
        # Stage 4: Baseline Model Training and Evaluation
        self._train_baseline_model()
        
        # Stage 5: Hyperparameter Optimization (if enabled)
        if self.config.hyperparam_enabled:
            self._optimize_hyperparameters()
        else:
            self.logger.info("Hyperparameter optimization disabled in config")
            self.optimized_results = self.baseline_results.copy()
        
        # Stage 6: Final Model Training with Best Parameters
        self._train_final_model()
        
        # Stage 7: Generate Artifacts
        self._generate_artifacts()
        
        # Stage 8: Generate Visualizations
        self._generate_visualizations()
        
        self.logger.info("\n" + "="*80)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("="*80)
        self.logger.info(f"Outputs saved to: {self.config.results_dir}")
        self.logger.info(f"Artifacts saved to: {self.config.artifacts_dir}")
        self.logger.info("="*80 + "\n")
    
    def _load_data(self):
        """Stage 1: Load raw data"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 1: DATA INGESTION")
        self.logger.info("="*80)
        
        self.logger.info(f"Loading data from:")
        self.logger.info(f"  - Demand: {self.config.demand_path}")
        self.logger.info(f"  - Plants: {self.config.plants_path}")
        self.logger.info(f"  - Costs: {self.config.costs_path}")
        
        self.demand_df = pd.read_csv(self.config.demand_path, keep_default_na=False, na_values=[""])
        self.plants_df = pd.read_csv(self.config.plants_path, keep_default_na=False, na_values=[""])
        self.costs_df = pd.read_csv(self.config.costs_path, keep_default_na=False, na_values=[""])
        
        self.logger.info(f"Data loaded successfully:")
        self.logger.info(f"  - Demand records: {len(self.demand_df)}")
        self.logger.info(f"  - Plant records: {len(self.plants_df)}")
        self.logger.info(f"  - Cost records: {len(self.costs_df)}")
    
    def _preprocess_data(self):
        """Stage 2: Preprocess data"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 2: PREPROCESSING")
        self.logger.info("="*80)
        
        # Initialize and fit preprocessor
        self.preprocessor = NECPreprocessor(
            plant_filter_percentile=self.config.plant_filter_percentile,
            missing_value_strategy=self.config.missing_value_strategy,
            random_seed=self.config.random_seed
        )
        # Fit on full data
        assert self.demand_df is not None and self.plants_df is not None and self.costs_df is not None, "Data must be loaded before preprocessing"
        data_tuple = cast(tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], (self.demand_df, self.plants_df, self.costs_df))
        self.preprocessor.fit(data_tuple)
        
        # Transform
        self.demand_df, self.plants_df, self.costs_df = self.preprocessor.transform(data_tuple)
        
        
        # Log summary
        summary = self.preprocessor.get_preprocessing_summary()
        self.logger.info(f"Preprocessing completed:")
        self.logger.info(f"  - Demand features: {summary['demand_features_count']}")
        self.logger.info(f"  - Plant features: {summary['plant_features_count']}")
        self.logger.info(f"  - Plants filtered: {summary['filtered_plants_count']}/{summary['total_plants_before']}")
        self.logger.info(f"  - Good plants retained: {summary['good_plants_count']}")
        self.logger.info(f"  - Final cost records: {len(self.costs_df)}")
    
    def _create_combined_dataset(self):
        """Stage 3: Create combined dataset"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 3: COMBINED DATASET CREATION")
        self.logger.info("="*80)
        
        assert self.demand_df is not None and self.plants_df is not None and self.costs_df is not None, "Data must be preprocessed before combining"
        assert self.preprocessor is not None, "Preprocessor must be initialized before creating combined dataset"
        assert self.preprocessor.demand_features_ is not None and self.preprocessor.plant_features_ is not None, "Preprocessor features must be fitted before creating combined dataset"
        
        self.combined_df = self.evaluator.create_combined_dataset(
            self.demand_df,
            self.plants_df,
            self.costs_df,
            self.preprocessor.demand_features_,
            self.preprocessor.plant_features_
        )
    
    def _train_baseline_model(self):
        """Stage 4: Train and evaluate baseline model"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 4: BASELINE MODEL TRAINING & EVALUATION")
        self.logger.info("="*80)
        
        # Validate preprocessor is initialized and fitted
        assert self.preprocessor is not None, "Preprocessor must be initialized before training baseline model"
        assert self.preprocessor.demand_features_ is not None, "Preprocessor demand_features_ must be fitted before training baseline model"
        assert self.preprocessor.plant_features_ is not None, "Preprocessor plant_features_ must be fitted before training baseline model"
        assert self.combined_df is not None, "Combined dataset must be created before training baseline model"
        
        # Create model with default parameters
        model_class = ModelFactory.create_model(
            self.config.model_type,
            self.config.model_params,
            self.config.random_seed
        ).__class__
        
        # Grouped train/test split
        X_train, X_test, y_train, y_test, test_demand_ids, combined_df_test = self.evaluator.grouped_train_test_split(
                self.combined_df,
                self.config.test_size_percent,
                self.preprocessor.demand_features_,
                self.preprocessor.plant_features_
            )
        
        # Train baseline model
        self.logger.info("Training baseline model...")
        baseline_model = ModelFactory.create_model(
            self.config.model_type,
            self.config.model_params,
            self.config.random_seed
        )
        baseline_model.fit(X_train, y_train)
        
        # Evaluate on test set
        test_errors, test_stats, scenario_table = self.evaluator.evaluate_on_test_set(
            baseline_model,
            X_test,
            y_test,
            combined_df_test,
            self.preprocessor.demand_features_,
            self.preprocessor.plant_features_
        )
        
        # LOGO Cross-Validation
        self.logger.info("\nPerforming LOGO Cross-Validation (baseline model)...")
        logo_errors, logo_fold_results = self.evaluator.logo_cross_validation(
            model_class,
            self.config.model_params,
            self.combined_df,
            self.preprocessor.demand_features_,
            self.preprocessor.plant_features_,
            n_folds=self.config.cv_n_folds,
            n_jobs=self.config.cv_n_jobs
        )
        
        logo_stats = calculate_error_statistics(np.array(logo_errors))
        
        # Store baseline results
        self.baseline_results = {
            'model': baseline_model,
            'params': self.config.model_params,
            'test_errors': test_errors,
            'test_stats': test_stats,
            'scenario_table': scenario_table,
            'logo_errors': logo_errors,
            'logo_stats': logo_stats,
            'logo_fold_results': logo_fold_results,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'combined_df_test': combined_df_test
        }
        
        self.logger.info(f"\nBaseline Model Results:")
        self.logger.info(f"  - Test Set RMSE: ${test_stats['rmse']:.2f}/MWh")
        self.logger.info(f"  - LOGO CV RMSE: ${logo_stats['rmse']:.2f}/MWh")
    
    def _optimize_hyperparameters(self):
        """Stage 5: Hyperparameter optimization"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 5: HYPERPARAMETER OPTIMIZATION")
        self.logger.info("="*80)
        
        # Validate preprocessor is initialized and fitted
        assert self.preprocessor is not None, "Preprocessor must be initialized before training baseline model"
        assert self.preprocessor.demand_features_ is not None, "Preprocessor demand_features_ must be fitted before training baseline model"
        assert self.preprocessor.plant_features_ is not None, "Preprocessor plant_features_ must be fitted before training baseline model"
        assert self.combined_df is not None, "Combined dataset must be created before training baseline model"
        
        model_class = ModelFactory.create_model(
            self.config.model_type,
            self.config.model_params,
            self.config.random_seed
        ).__class__
        
        best_params, best_rmse, leaderboard = self.optimizer.optimize(
            model_class,
            self.config.tuning_grid,
            self.combined_df,
            self.preprocessor.demand_features_,
            self.preprocessor.plant_features_,
            n_cv_folds=self.config.hyperparam_cv_folds
        )
        
        # Train optimized model on full training set and evaluate
        self.logger.info("\nTraining optimized model on full training set...")
        optimized_model = ModelFactory.create_model(
            self.config.model_type,
            best_params,
            self.config.random_seed
        )
        optimized_model.fit(
            self.baseline_results['X_train'],
            self.baseline_results['y_train']
        )
        
        # Evaluate optimized model
        opt_test_errors, opt_test_stats, opt_scenario_table = self.evaluator.evaluate_on_test_set(
            optimized_model,
            self.baseline_results['X_test'],
            self.baseline_results['y_test'],
            self.baseline_results['combined_df_test'],
            self.preprocessor.demand_features_,
            self.preprocessor.plant_features_
        )
        
        # LOGO CV for optimized model
        self.logger.info("\nPerforming LOGO Cross-Validation (optimized model)...")
        opt_logo_errors, opt_logo_fold_results = self.evaluator.logo_cross_validation(
            model_class,
            best_params,
            self.combined_df,
            self.preprocessor.demand_features_,
            self.preprocessor.plant_features_,
            n_folds=self.config.cv_n_folds,
            n_jobs=self.config.cv_n_jobs
        )
        
        opt_logo_stats = calculate_error_statistics(np.array(opt_logo_errors))
        
        # Store optimized results
        self.optimized_results = {
            'model': optimized_model,
            'params': best_params,
            'test_errors': opt_test_errors,
            'test_stats': opt_test_stats,
            'scenario_table': opt_scenario_table,
            'logo_errors': opt_logo_errors,
            'logo_stats': opt_logo_stats,
            'logo_fold_results': opt_logo_fold_results,
            'leaderboard': leaderboard,
            'best_rmse': best_rmse
        }
        
        improvement = ((self.baseline_results['test_stats']['rmse'] - opt_test_stats['rmse']) / 
                      self.baseline_results['test_stats']['rmse'] * 100)
        
        self.logger.info(f"\nOptimized Model Results:")
        self.logger.info(f"  - Best Parameters: {best_params}")
        self.logger.info(f"  - Test Set RMSE: ${opt_test_stats['rmse']:.2f}/MWh")
        self.logger.info(f"  - LOGO CV RMSE: ${opt_logo_stats['rmse']:.2f}/MWh")
        self.logger.info(f"  - Improvement over baseline: {improvement:.2f}%")
    
    def _train_final_model(self):
        """Stage 6: Train final model with best parameters"""
        
    
    def _generate_artifacts(self):
        """Stage 7: Generate all output artifacts"""
    
    
    def _generate_visualizations(self):
        """Stage 8: Generate EDA and results visualizations"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 8: GENERATING VISUALIZATIONS")
        self.logger.info("="*80)
        
        # Create visualizer
        viz = Visualizer(output_dir=f"{self.config.results_dir}/plots")
        

        
        viz.plot_cost_distribution(self.costs_df)
        viz.plot_plant_performance(self.costs_df)
        
        
        