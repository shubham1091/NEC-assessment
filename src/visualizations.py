"""
EDA and Visualization Module
Creates plots for understanding data and model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class Visualizer:
    """Creates EDA and results visualizations"""
    
    def __init__(self, output_dir='outputs/plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_cost_distribution(self, costs_df):
        """Plot cost distribution across plants"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall cost distribution
        ax1.hist(costs_df['Cost_USD_per_MWh'], bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Cost (USD/MWh)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Cost Distribution')
        ax1.axvline(costs_df['Cost_USD_per_MWh'].mean(), color='red', 
                   linestyle='--', label=f'Mean: ${costs_df["Cost_USD_per_MWh"].mean():.2f}')
        ax1.legend()
        
        # Cost by plant (box plot for top 10 plants by median cost)
        plant_costs = costs_df.groupby('Plant ID')['Cost_USD_per_MWh'].median().sort_values()
        top_plants = plant_costs.head(10).index
        
        data_to_plot = [costs_df[costs_df['Plant ID'] == p]['Cost_USD_per_MWh'].values 
                        for p in top_plants]
        ax2.boxplot(data_to_plot, labels=top_plants)
        ax2.set_xlabel('Plant ID')
        ax2.set_ylabel('Cost (USD/MWh)')
        ax2.set_title('Cost Distribution - Top 10 Best Plants')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cost_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / 'cost_distribution.png'}")
    
    def plot_plant_performance(self, costs_df):
        """Plot plant performance summary"""
        plant_stats = costs_df.groupby('Plant ID')['Cost_USD_per_MWh'].agg(['mean', 'median', 'std', 'count'])
        plant_stats = plant_stats.sort_values('median')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot median costs with error bars (std)
        x = range(len(plant_stats))
        ax.errorbar(x, plant_stats['median'], yerr=plant_stats['std'], 
                   fmt='o', capsize=3, alpha=0.6, label='Median ± Std')
        ax.plot(x, plant_stats['mean'], 's', alpha=0.4, label='Mean', markersize=3)
        
        ax.set_xlabel('Plant (sorted by median cost)')
        ax.set_ylabel('Cost (USD/MWh)')
        ax.set_title('Plant Performance Summary')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plant_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / 'plant_performance.png'}")
    
    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance for tree-based models"""
        if not hasattr(model, 'feature_importances_'):
            print("Model doesn't have feature_importances_. Skipping.")
            return
        
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:20]  # Top 20
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / 'feature_importance.png'}")
    
    def plot_predictions_analysis(self, y_true, y_pred, title='Model Predictions'):
        """Plot actual vs predicted and residuals"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Actual vs Predicted
        ax1.scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Cost (USD/MWh)')
        ax1.set_ylabel('Predicted Cost (USD/MWh)')
        ax1.set_title(f'{title}: Actual vs Predicted')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_pred - y_true
        ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', label='Zero Error')
        ax2.set_xlabel('Residual (Predicted - Actual)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{title}: Residual Distribution')
        ax2.legend()
        
        plt.tight_layout()
        filename = title.lower().replace(' ', '_') + '.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / filename}")
    
    def plot_selection_errors(self, scenario_table):
        """Plot selection error distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        errors = scenario_table['Selection_Error'].values
        
        # Error distribution
        ax1.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', label='Zero Error (Perfect Selection)')
        ax1.axvline(errors.mean(), color='green', linestyle='--', 
                   label=f'Mean Error: ${errors.mean():.2f}')
        ax1.set_xlabel('Selection Error (USD/MWh)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Plant Selection Error Distribution')
        ax1.legend()
        
        # Success rate (zero error)
        total = len(errors)
        perfect = (errors == 0).sum()
        near_perfect = (errors <= 1).sum()
        
        categories = ['Perfect\n(error=0)', 'Near Perfect\n(error≤1)', 'Suboptimal\n(error>1)']
        counts = [perfect, near_perfect - perfect, total - near_perfect]
        colors = ['green', 'yellow', 'red']
        
        ax2.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Number of Demands')
        ax2.set_title('Plant Selection Quality')
        
        # Add percentage labels
        for i, (cat, count) in enumerate(zip(categories, counts)):
            pct = 100 * count / total
            ax2.text(i, count + 0.5, f'{count}\n({pct:.1f}%)', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'selection_errors.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / 'selection_errors.png'}")
    
    def plot_cv_results(self, logo_fold_results):
        """Plot LOGO CV fold results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        errors = logo_fold_results['Selection_Error'].values
        
        # Error over folds
        ax1.plot(logo_fold_results['Fold'], errors, alpha=0.6, linewidth=0.5)
        ax1.axhline(0, color='red', linestyle='--', alpha=0.5, label='Zero Error')
        ax1.axhline(errors.mean(), color='green', linestyle='--', 
                   label=f'Mean: ${errors.mean():.2f}')
        ax1.set_xlabel('Fold Number')
        ax1.set_ylabel('Selection Error (USD/MWh)')
        ax1.set_title('LOGO CV: Error Across Folds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative RMSE
        cumulative_rmse = []
        for i in range(1, len(errors) + 1):
            rmse = np.sqrt(np.mean(errors[:i] ** 2))
            cumulative_rmse.append(rmse)
        
        ax2.plot(range(1, len(errors) + 1), cumulative_rmse)
        ax2.set_xlabel('Number of Folds')
        ax2.set_ylabel('Cumulative RMSE (USD/MWh)')
        ax2.set_title('LOGO CV: Cumulative RMSE Convergence')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cv_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / 'cv_results.png'}")
    
