#!/usr/bin/env python3
"""
Vegetation Degradation Analysis Tool

This script processes predicted vs observed vegetation degradation data
and calculates accuracy metrics by region.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class VegetationDegradationAnalyzer:
    """Analyzes predicted vs observed vegetation degradation by region."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the analyzer.
        
        Args:
            data_path: Path to the CSV file containing the data
        """
        self.data = None
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path: str):
        """
        Load vegetation degradation data from CSV file.
        
        Expected columns: region, predicted, observed
        
        Args:
            data_path: Path to the CSV file
        """
        self.data = pd.read_csv(data_path)
        required_cols = ['region', 'predicted', 'observed']
        
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
    
    def set_data(self, data: pd.DataFrame):
        """
        Set data directly from a DataFrame.
        
        Args:
            data: DataFrame with columns: region, predicted, observed
        """
        required_cols = ['region', 'predicted', 'observed']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        self.data = data
    
    def calculate_metrics(self) -> pd.DataFrame:
        """
        Calculate accuracy metrics for each region.
        
        Returns:
            DataFrame with metrics by region
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() or set_data() first.")
        
        metrics = []
        
        for region in self.data['region'].unique():
            region_data = self.data[self.data['region'] == region]
            predicted = region_data['predicted'].values
            observed = region_data['observed'].values
            
            # Calculate various accuracy metrics
            mae = np.mean(np.abs(predicted - observed))
            rmse = np.sqrt(np.mean((predicted - observed) ** 2))
            
            # Correlation coefficient
            correlation = np.corrcoef(predicted, observed)[0, 1]
            
            # R-squared
            ss_res = np.sum((observed - predicted) ** 2)
            ss_tot = np.sum((observed - np.mean(observed)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((observed - predicted) / (observed + 1e-10))) * 100
            
            metrics.append({
                'region': region,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r_squared,
                'Correlation': correlation,
                'MAPE': mape,
                'n_samples': len(region_data)
            })
        
        return pd.DataFrame(metrics)
    
    def plot_predictions(self, save_path: str = None):
        """
        Create scatter plots of predicted vs observed values by region.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() or set_data() first.")
        
        regions = self.data['region'].unique()
        n_regions = len(regions)
        
        # Create subplot grid
        n_cols = min(3, n_regions)
        n_rows = (n_regions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_regions == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_regions > 1 else [axes]
        
        for idx, region in enumerate(regions):
            ax = axes[idx]
            region_data = self.data[self.data['region'] == region]
            
            # Scatter plot
            ax.scatter(region_data['observed'], region_data['predicted'], 
                      alpha=0.6, edgecolors='k', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(region_data['observed'].min(), region_data['predicted'].min())
            max_val = max(region_data['observed'].max(), region_data['predicted'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', label='Perfect Prediction', linewidth=2)
            
            # Calculate R²
            predicted = region_data['predicted'].values
            observed = region_data['observed'].values
            ss_res = np.sum((observed - predicted) ** 2)
            ss_tot = np.sum((observed - np.mean(observed)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            ax.set_xlabel('Observed Degradation', fontsize=10)
            ax.set_ylabel('Predicted Degradation', fontsize=10)
            ax.set_title(f'{region}\n(R² = {r_squared:.3f}, n = {len(region_data)})', 
                        fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_regions, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def generate_report(self) -> str:
        """
        Generate a text report of the analysis.
        
        Returns:
            Formatted report string
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() or set_data() first.")
        
        metrics_df = self.calculate_metrics()
        
        report = "=" * 80 + "\n"
        report += "VEGETATION DEGRADATION ACCURACY REPORT\n"
        report += "Predicted vs Observed by Region\n"
        report += "=" * 80 + "\n\n"
        
        for _, row in metrics_df.iterrows():
            report += f"Region: {row['region']}\n"
            report += f"  Number of samples: {row['n_samples']}\n"
            report += f"  Mean Absolute Error (MAE): {row['MAE']:.4f}\n"
            report += f"  Root Mean Square Error (RMSE): {row['RMSE']:.4f}\n"
            report += f"  R-squared (R²): {row['R2']:.4f}\n"
            report += f"  Correlation: {row['Correlation']:.4f}\n"
            report += f"  Mean Absolute Percentage Error (MAPE): {row['MAPE']:.2f}%\n"
            report += "\n"
        
        report += "=" * 80 + "\n"
        
        return report


def main():
    """Main function for command-line usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python process_ellipses.py <data_file.csv>")
        print("\nExpected CSV format:")
        print("region,predicted,observed")
        print("Region1,0.45,0.50")
        print("Region1,0.62,0.58")
        print("...")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    # Create analyzer
    analyzer = VegetationDegradationAnalyzer(data_path)
    
    # Generate and print report
    report = analyzer.generate_report()
    print(report)
    
    # Calculate and display metrics table
    metrics = analyzer.calculate_metrics()
    print("\nDetailed Metrics Table:")
    print(metrics.to_string(index=False))
    
    # Create and save plot
    output_plot = data_path.replace('.csv', '_plot.png')
    analyzer.plot_predictions(save_path=output_plot)
    print(f"\nVisualization saved to: {output_plot}")


if __name__ == "__main__":
    main()
