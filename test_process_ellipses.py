#!/usr/bin/env python3
"""
Tests for the Vegetation Degradation Analyzer
"""

import pandas as pd
import numpy as np
import os
import sys
from process_ellipses import VegetationDegradationAnalyzer


def test_load_data():
    """Test loading data from CSV."""
    print("Testing data loading...")
    analyzer = VegetationDegradationAnalyzer('example_data.csv')
    assert analyzer.data is not None
    assert 'region' in analyzer.data.columns
    assert 'predicted' in analyzer.data.columns
    assert 'observed' in analyzer.data.columns
    print("✓ Data loading test passed")


def test_set_data():
    """Test setting data directly."""
    print("Testing direct data setting...")
    data = pd.DataFrame({
        'region': ['A', 'A', 'B', 'B'],
        'predicted': [0.5, 0.6, 0.4, 0.7],
        'observed': [0.52, 0.58, 0.42, 0.68]
    })
    analyzer = VegetationDegradationAnalyzer()
    analyzer.set_data(data)
    assert analyzer.data is not None
    assert len(analyzer.data) == 4
    print("✓ Direct data setting test passed")


def test_calculate_metrics():
    """Test metric calculation."""
    print("Testing metric calculation...")
    data = pd.DataFrame({
        'region': ['A', 'A', 'A', 'B', 'B', 'B'],
        'predicted': [0.5, 0.6, 0.7, 0.4, 0.5, 0.6],
        'observed': [0.52, 0.58, 0.68, 0.42, 0.48, 0.62]
    })
    analyzer = VegetationDegradationAnalyzer()
    analyzer.set_data(data)
    metrics = analyzer.calculate_metrics()
    
    assert 'region' in metrics.columns
    assert 'MAE' in metrics.columns
    assert 'RMSE' in metrics.columns
    assert 'R2' in metrics.columns
    assert 'Correlation' in metrics.columns
    assert 'MAPE' in metrics.columns
    assert len(metrics) == 2  # Two regions
    print("✓ Metric calculation test passed")


def test_generate_report():
    """Test report generation."""
    print("Testing report generation...")
    analyzer = VegetationDegradationAnalyzer('example_data.csv')
    report = analyzer.generate_report()
    
    assert 'VEGETATION DEGRADATION ACCURACY REPORT' in report
    assert 'Region:' in report
    assert 'MAE' in report or 'Mean Absolute Error' in report
    print("✓ Report generation test passed")


def test_plot_predictions():
    """Test visualization creation."""
    print("Testing visualization creation...")
    analyzer = VegetationDegradationAnalyzer('example_data.csv')
    test_plot_path = '/tmp/test_plot.png'
    
    # Clean up any existing test plot
    if os.path.exists(test_plot_path):
        os.remove(test_plot_path)
    
    fig = analyzer.plot_predictions(save_path=test_plot_path)
    
    assert os.path.exists(test_plot_path)
    assert os.path.getsize(test_plot_path) > 0
    
    # Clean up
    os.remove(test_plot_path)
    print("✓ Visualization creation test passed")


def test_accuracy_metrics():
    """Test accuracy of metric calculations."""
    print("Testing metric accuracy...")
    # Create perfect prediction case
    data = pd.DataFrame({
        'region': ['A'] * 5,
        'predicted': [1, 2, 3, 4, 5],
        'observed': [1, 2, 3, 4, 5]
    })
    analyzer = VegetationDegradationAnalyzer()
    analyzer.set_data(data)
    metrics = analyzer.calculate_metrics()
    
    # For perfect predictions
    assert abs(metrics.iloc[0]['MAE']) < 1e-10
    assert abs(metrics.iloc[0]['RMSE']) < 1e-10
    assert abs(metrics.iloc[0]['Correlation'] - 1.0) < 1e-10
    print("✓ Metric accuracy test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Vegetation Degradation Analyzer Tests")
    print("="*60 + "\n")
    
    try:
        test_load_data()
        test_set_data()
        test_calculate_metrics()
        test_generate_report()
        test_plot_predictions()
        test_accuracy_metrics()
        
        print("\n" + "="*60)
        print("✓ All tests passed successfully!")
        print("="*60 + "\n")
        return True
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
