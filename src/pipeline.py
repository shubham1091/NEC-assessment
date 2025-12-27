"""
Main NEC ML Pipeline orchestrator.
Coordinates all pipeline stages from data ingestion to artifact generation.

Team Contribution: All team members - Integrated pipeline design
"""

import pandas as pd
import numpy as np

import logging
from pathlib import Path
from datetime import datetime

from .config import Config


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
        self.evaluator = None
        self.optimizer = None
        
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
        
    
    def _create_combined_dataset(self):
        """Stage 3: Create combined dataset"""
        
    
    def _train_baseline_model(self):
        """Stage 4: Train and evaluate baseline model"""
        
    
    def _optimize_hyperparameters(self):
        """Stage 5: Hyperparameter optimization"""
        
    
    def _train_final_model(self):
        """Stage 6: Train final model with best parameters"""
        
    
    def _generate_artifacts(self):
        """Stage 7: Generate all output artifacts"""
    
    
    def _generate_visualizations(self):
        """Stage 8: Generate EDA and results visualizations"""
        
