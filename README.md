# process_ellipses
Predicted vs Observed Vegetation Degradation Accuracy by Region

## Overview

This tool analyzes predicted vs observed vegetation degradation data and calculates accuracy metrics by region. It provides comprehensive statistical analysis and visualizations to evaluate model performance across different geographical regions.

## Features

- **Multiple Accuracy Metrics**: Calculates MAE, RMSE, R², Correlation, and MAPE
- **Regional Analysis**: Compare model performance across different regions
- **Visualizations**: Generate scatter plots showing predicted vs observed values
- **Easy to Use**: Simple command-line interface and Python API

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python process_ellipses.py <data_file.csv>
```

Example:
```bash
python process_ellipses.py example_data.csv
```

### Python API

```python
from process_ellipses import VegetationDegradationAnalyzer

# Create analyzer
analyzer = VegetationDegradationAnalyzer('your_data.csv')

# Generate report
report = analyzer.generate_report()
print(report)

# Get metrics as DataFrame
metrics = analyzer.calculate_metrics()
print(metrics)

# Create visualization
analyzer.plot_predictions(save_path='output_plot.png')
```

## Data Format

The input CSV file should have the following columns:

| Column | Description |
|--------|-------------|
| region | Name of the geographical region |
| predicted | Predicted vegetation degradation value |
| observed | Observed vegetation degradation value |

Example:
```csv
region,predicted,observed
North,0.45,0.50
North,0.62,0.58
South,0.32,0.35
South,0.48,0.44
```

## Metrics Explained

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and observed values
- **RMSE (Root Mean Square Error)**: Square root of average squared differences
- **R² (R-squared)**: Proportion of variance in observed values explained by predictions
- **Correlation**: Linear correlation coefficient between predicted and observed values
- **MAPE (Mean Absolute Percentage Error)**: Average percentage difference

## Example Output

```
================================================================================
VEGETATION DEGRADATION ACCURACY REPORT
Predicted vs Observed by Region
================================================================================

Region: North
  Number of samples: 10
  Mean Absolute Error (MAE): 0.0340
  Root Mean Square Error (RMSE): 0.0349
  R-squared (R²): 0.8093
  Correlation: 0.9567
  Mean Absolute Percentage Error (MAPE): 6.48%
...
```

The tool also generates visualization plots showing predicted vs observed values for each region.

## License

MIT
