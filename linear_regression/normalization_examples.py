"""
Examples and Usage Guide for Normalization Handler
=================================================

This file demonstrates how to use the NormalizationHandler class
for various data preprocessing tasks in machine learning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from normalization_handler import NormalizationHandler, quick_minmax, quick_standardize

def basic_usage_example():
    """Basic usage examples of the NormalizationHandler."""
    print("=== Basic Usage Examples ===\n")
    
    # Create sample data
    data = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    print(f"Original data: {data}")
    print(f"Mean: {np.mean(data):.2f}, Std: {np.std(data):.2f}")
    print(f"Min: {np.min(data)}, Max: {np.max(data)}\n")
    
    # Initialize normalizer
    normalizer = NormalizationHandler()
    # Spiritualize the data and making it more suitable for certain models 
    spiritualized = normalizer.spiritualize(data)
    print(f"Spiritualized: {spiritualized}")        
    print(f"Range: [{np.min(spiritualized):.3f}, {np.max(spiritualized):.3f}]\n")
    
    # Min-Max Normalization (0-1)
    minmax_norm = normalizer.min_max_normalize(data)
    print(f"Min-Max (0-1): {minmax_norm}")
    print(f"Range: [{np.min(minmax_norm):.3f}, {np.max(minmax_norm):.3f}]\n")
    # update
    # Min-Max Normalization (-1 to 1)
    minmax_custom = normalizer.min_max_normalize(data, feature_range=(-1, 1))
    print(f"Min-Max (-1,1): {minmax_custom}")
    print(f"Range: [{np.min(minmax_custom):.3f}, {np.max(minmax_custom):.3f}]\n")
    
    # Standardization
    standardized = normalizer.standardize(data)
    print(f"Standardized: {standardized}")
    print(f"Mean: {np.mean(standardized):.6f}, Std: {np.std(standardized):.6f}\n")
    
    # Robust Scaling
    robust_scaled = normalizer.robust_scale(data)  # Use default quantiles
    print(f"Robust Scaled: {robust_scaled}")
    print(f"Median: {np.median(robust_scaled):.6f}\n")

def dataframe_example():
    """Example using pandas DataFrame."""
    print("=== DataFrame Example ===\n")
    
    # Create sample DataFrame
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 50),
        'income': np.random.randint(20000, 150000, 50),
        'score': np.random.uniform(0, 100, 50)
    })
    
    print("Original DataFrame statistics:")
    print(df.describe())
    print()
    
    # Initialize normalizer
    normalizer = NormalizationHandler()
    
    # Normalize different columns with different methods
    df_normalized = df.copy()
    df_normalized['age'] = normalizer.min_max_normalize(df[['age']])
    df_normalized['income'] = normalizer.standardize(df[['income']], fit=False)  # Use new scaler
    df_normalized['score'] = normalizer.robust_scale(df[['score']], fit=False)   # Use new scaler
    
    print("Normalized DataFrame statistics:")
    print(df_normalized.describe())
    print()

def train_test_split_example():
    """Example showing proper train/test normalization."""
    print("=== Train/Test Split Example ===\n")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 3) * 10 + 50
    
    # Split into train/test
    train_size = int(0.8 * len(X))
    X_train = X[:train_size]
    X_test = X[train_size:]
    
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Train mean: {np.mean(X_train, axis=0)}")
    print(f"Test mean: {np.mean(X_test, axis=0)}\n")
    
    # Initialize normalizer
    normalizer = NormalizationHandler()
    
    # Fit on training data only
    X_train_norm = normalizer.standardize(X_train, fit=True)
    
    # Transform test data using training parameters
    X_test_norm = normalizer.standardize(X_test, fit=False)
    
    print("After normalization:")
    print(f"Train mean: {np.mean(X_train_norm, axis=0)}")
    print(f"Train std: {np.std(X_train_norm, axis=0)}")
    print(f"Test mean: {np.mean(X_test_norm, axis=0)}")
    print(f"Test std: {np.std(X_test_norm, axis=0)}\n")
    
    # Inverse transform to verify
    X_train_recovered = normalizer.inverse_transform(X_train_norm, 'standard')
    print(f"Recovery error: {np.max(np.abs(X_train - X_train_recovered))}")

def outlier_handling_example():
    """Example showing how different methods handle outliers."""
    print("=== Outlier Handling Example ===\n")
    
    # Create data with outliers
    np.random.seed(42)
    normal_data = np.random.randn(95) * 2 + 10
    outliers = np.array([50, -20, 45, -15, 40])  # Extreme outliers
    data_with_outliers = np.concatenate([normal_data, outliers])
    
    print(f"Data with outliers - Mean: {np.mean(data_with_outliers):.2f}, Std: {np.std(data_with_outliers):.2f}")
    print(f"Min: {np.min(data_with_outliers):.2f}, Max: {np.max(data_with_outliers):.2f}\n")
    
    normalizer = NormalizationHandler()
    
    # Compare different normalization methods
    minmax_norm = normalizer.min_max_normalize(data_with_outliers)
    standard_norm = normalizer.standardize(data_with_outliers, fit=False)
    robust_norm = normalizer.robust_scale(data_with_outliers, fit=False)
    
    print("Normalization results:")
    print(f"Min-Max - Range: [{np.min(minmax_norm):.3f}, {np.max(minmax_norm):.3f}]")
    print(f"Standard - Mean: {np.mean(standard_norm):.3f}, Std: {np.std(standard_norm):.3f}")
    print(f"Robust - Median: {np.median(robust_norm):.3f}, IQR: {np.percentile(robust_norm, 75) - np.percentile(robust_norm, 25):.3f}")
    print()
    
    # Show how outliers affect each method
    print("Effect of outliers on normal data points:")
    normal_indices = list(range(95))  # Indices of normal data points
    
    print(f"Min-Max normal data range: [{np.min(minmax_norm[normal_indices]):.3f}, {np.max(minmax_norm[normal_indices]):.3f}]")
    print(f"Standard normal data range: [{np.min(standard_norm[normal_indices]):.3f}, {np.max(standard_norm[normal_indices]):.3f}]")
    print(f"Robust normal data range: [{np.min(robust_norm[normal_indices]):.3f}, {np.max(robust_norm[normal_indices]):.3f}]")

def quick_functions_example():
    """Example using quick utility functions."""
    print("=== Quick Functions Example ===\n")
    
    data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    print(f"Original data: {data}")
    
    # Quick normalization without storing scalers
    quick_minmax_result = quick_minmax(data)
    quick_standard_result = quick_standardize(data)
    
    print(f"Quick Min-Max: {quick_minmax_result}")
    print(f"Quick Standardize: {quick_standard_result}")
    
    # Custom range
    quick_custom = quick_minmax(data, feature_range=(-10, 10))
    print(f"Quick Custom Range (-10, 10): {quick_custom}")

def log_normalization_example():
    """Example of logarithmic normalization."""
    print("=== Log Normalization Example ===\n")
    
    # Create skewed data (e.g., income distribution)
    np.random.seed(42)
    skewed_data = np.random.exponential(scale=2, size=100) * 10000
    
    print(f"Original skewed data - Mean: {np.mean(skewed_data):.2f}, Std: {np.std(skewed_data):.2f}")
    print(f"Skewness: {np.mean(((skewed_data - np.mean(skewed_data)) / np.std(skewed_data))**3):.2f}")
    
    normalizer = NormalizationHandler()
    
    # Apply log normalization
    log_normalized = normalizer.log_normalize(skewed_data, base='natural')
    
    print(f"Log normalized - Mean: {np.mean(log_normalized):.2f}, Std: {np.std(log_normalized):.2f}")
    print(f"Skewness after log: {np.mean(((log_normalized - np.mean(log_normalized)) / np.std(log_normalized))**3):.2f}")

def main():
    """Run all examples."""
    basic_usage_example()
    print("\n" + "="*60 + "\n")
    
    dataframe_example()
    print("\n" + "="*60 + "\n")
    
    train_test_split_example()
    print("\n" + "="*60 + "\n")
    
    outlier_handling_example()
    print("\n" + "="*60 + "\n")
    
    quick_functions_example()
    print("\n" + "="*60 + "\n")
    
    log_normalization_example()

if __name__ == "__main__":
    main()
