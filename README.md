# NEC-assessment

**Advanced Data Machine Learning (ADML) Group 5 Project**

## Team Members
- Shubham Verma
- Aisosa Erhunmwunsee
- Krishna Prasanth Reenamol
- Leo Makori
- Waleed Asad

## Project Overview

This project implements a machine learning pipeline for NEC plant selection optimization. It uses Random Forest models to predict generation costs and optimize plant selection based on demand scenarios.

## Prerequisites

- Python 3.8+
- Conda (recommended for environment management)
- Required packages listed in environment setup

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/shubham1091/NEC-assessment.git
cd NEC-assessment
```

### 3. Prepare Data

Ensure the following data files are in `data/raw/`:
- `demand.csv` - Demand scenarios
- `plants.csv` - Plant information
- `generation_costs.csv` - Generation cost data

## Running the Code

### Option 1: Run the Complete Pipeline
```bash
python run_pipeline.py
```

This will execute all stages:
1. Data ingestion
2. Preprocessing
3. Combined dataset creation
4. Baseline model training & evaluation
5. Hyperparameter optimization
6. Final model training
7. Artifact generation
8. Visualization generation


### Option 2: Use Trained Model
```bash
python use_model.py
```

This loads the pre-trained model from `artifacts/` and makes predictions on new data.

## Configuration

Edit `config.yaml` to customize:
- Data paths
- Model hyperparameters
- Cross-validation settings
- Output directories

## Output Files

After running the pipeline, outputs are saved to:

### Results (`outputs/`)
- `model_comparison.md` - Detailed model performance report
- `hyperparameter_leaderboard.csv` - Top parameter combinations
- `logo_cv_folds.csv` - Cross-validation fold results
- `scenario_selections.csv` - Plant selections for each scenario
- `run_metadata.json` - Pipeline execution metadata
- `plots/` - Visualization outputs

### Artifacts (`artifacts/`)
- `final_model.pkl` - Trained Random Forest model
- `preprocessor.pkl` - Data preprocessor
- `config_effective.yaml` - Effective configuration used
- `feature_names.json` - Feature names for reference

### Logs (`logs/`)
- `pipeline_*.log` - Detailed pipeline execution logs
