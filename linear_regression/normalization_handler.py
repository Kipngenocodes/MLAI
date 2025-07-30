"""
Comprehensive Normalization Handler for Machine Learning
========================================================

This module provides various normalization techniques commonly used in machine learning
and data preprocessing. It includes methods for feature scaling, standardization,
and other normalization approaches.

Author: ML/AI Project
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import warnings

class NormalizationHandler:
    """
    A comprehensive class for handling various normalization techniques.
    
    This class provides methods for:
    - Min-Max Normalization (Feature Scaling)
    - Z-Score Standardization
    - Robust Scaling
    - Unit Vector Scaling
    - Custom range normalization
    - Batch normalization utilities
    """
    
    def __init__(self):
        """Initialize the NormalizationHandler."""
        self.scalers = {}
        self.fitted_methods = set()
        
    def min_max_normalize(self, data: Union[np.ndarray, pd.DataFrame], 
                         feature_range: Tuple[float, float] = (0, 1),
                         fit: bool = True) -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply Min-Max normalization to scale features to a given range.
        
        Formula: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
        
        Args:
            data: Input data to normalize
            feature_range: Desired range of transformed data (default: (0, 1))
            fit: Whether to fit the scaler or use existing fitted scaler
            
        Returns:
            Normalized data
        """
        if fit or 'minmax' not in self.scalers:
            self.scalers['minmax'] = MinMaxScaler(feature_range=feature_range)
            if isinstance(data, pd.DataFrame):
                normalized = pd.DataFrame(
                    self.scalers['minmax'].fit_transform(data),
                    columns=data.columns,
                    index=data.index
                )
            else:
                data_reshaped = data.reshape(-1, 1) if data.ndim == 1 else data
                normalized = self.scalers['minmax'].fit_transform(data_reshaped)
                if data.ndim == 1:
                    normalized = normalized.flatten()
            self.fitted_methods.add('minmax')
        else:
            if isinstance(data, pd.DataFrame):
                normalized = pd.DataFrame(
                    self.scalers['minmax'].transform(data),
                    columns=data.columns,
                    index=data.index
                )
            else:
                data_reshaped = data.reshape(-1, 1) if data.ndim == 1 else data
                normalized = self.scalers['minmax'].transform(data_reshaped)
                if data.ndim == 1:
                    normalized = normalized.flatten()
                    
        return normalized
    
    def standardize(self, data: Union[np.ndarray, pd.DataFrame], 
                   fit: bool = True) -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply Z-score standardization (mean=0, std=1).
        
        Formula: X_scaled = (X - mean) / std
        
        Args:
            data: Input data to standardize
            fit: Whether to fit the scaler or use existing fitted scaler
            
        Returns:
            Standardized data
        """
        if fit or 'standard' not in self.scalers:
            self.scalers['standard'] = StandardScaler()
            if isinstance(data, pd.DataFrame):
                standardized = pd.DataFrame(
                    self.scalers['standard'].fit_transform(data),
                    columns=data.columns,
                    index=data.index
                )
            else:
                data_reshaped = data.reshape(-1, 1) if data.ndim == 1 else data
                standardized = self.scalers['standard'].fit_transform(data_reshaped)
                if data.ndim == 1:
                    standardized = standardized.flatten()
            self.fitted_methods.add('standard')
        else:
            if isinstance(data, pd.DataFrame):
                standardized = pd.DataFrame(
                    self.scalers['standard'].transform(data),
                    columns=data.columns,
                    index=data.index
                )
            else:
                data_reshaped = data.reshape(-1, 1) if data.ndim == 1 else data
                standardized = self.scalers['standard'].transform(data_reshaped)
                if data.ndim == 1:
                    standardized = standardized.flatten()
                    
        return standardized
    
    def robust_scale(self, data: Union[np.ndarray, pd.DataFrame], 
                    fit: bool = True) -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply robust scaling using median and interquartile range.
        Less sensitive to outliers than standard scaling.
        
        Formula: X_scaled = (X - median) / IQR
        
        Args:
            data: Input data to scale
            fit: Whether to fit the scaler or use existing fitted scaler
            
        Returns:
            Robust scaled data
        """
        if fit or 'robust' not in self.scalers:
            self.scalers['robust'] = RobustScaler()
            if isinstance(data, pd.DataFrame):
                scaled = pd.DataFrame(
                    self.scalers['robust'].fit_transform(data),
                    columns=data.columns,
                    index=data.index
                )
            else:
                data_reshaped = data.reshape(-1, 1) if data.ndim == 1 else data
                scaled = self.scalers['robust'].fit_transform(data_reshaped)
                if data.ndim == 1:
                    scaled = scaled.flatten()
            self.fitted_methods.add('robust')
        else:
            if isinstance(data, pd.DataFrame):
                scaled = pd.DataFrame(
                    self.scalers['robust'].transform(data),
                    columns=data.columns,
                    index=data.index
                )
            else:
                data_reshaped = data.reshape(-1, 1) if data.ndim == 1 else data
                scaled = self.scalers['robust'].transform(data_reshaped)
                if data.ndim == 1:
                    scaled = scaled.flatten()
                    
        return scaled
    
    def unit_vector_normalize(self, data: Union[np.ndarray, pd.DataFrame], 
                             norm: str = 'l2') -> Union[np.ndarray, pd.DataFrame]:
        """
        Normalize samples individually to unit norm.
        
        Args:
            data: Input data to normalize
            norm: The norm to use ('l1', 'l2', or 'max')
            
        Returns:
            Unit normalized data
        """
        normalizer = Normalizer(norm=norm)
        
        if isinstance(data, pd.DataFrame):
            normalized = pd.DataFrame(
                normalizer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
        else:
            data_reshaped = data.reshape(1, -1) if data.ndim == 1 else data
            normalized = normalizer.fit_transform(data_reshaped)
            if data.ndim == 1:
                normalized = normalized.flatten()
                
        return normalized
    
    def custom_range_normalize(self, data: Union[np.ndarray, pd.DataFrame], 
                              target_min: float, target_max: float) -> Union[np.ndarray, pd.DataFrame]:
        """
        Normalize data to a custom range.
        
        Args:
            data: Input data to normalize
            target_min: Minimum value of target range
            target_max: Maximum value of target range
            
        Returns:
            Data normalized to custom range
        """
        if isinstance(data, pd.DataFrame):
            data_min = data.min()
            data_max = data.max()
            normalized = (data - data_min) / (data_max - data_min) * (target_max - target_min) + target_min
        else:
            data_min = np.min(data)
            data_max = np.max(data)
            normalized = (data - data_min) / (data_max - data_min) * (target_max - target_min) + target_min
            
        return normalized
    
    def log_normalize(self, data: Union[np.ndarray, pd.DataFrame], 
                     base: str = 'natural') -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply logarithmic normalization.
        
        Args:
            data: Input data to normalize (must be positive)
            base: Logarithm base ('natural', '10', '2')
            
        Returns:
            Log normalized data
        """
        if isinstance(data, pd.DataFrame):
            if (data <= 0).any().any():
                warnings.warn("Data contains non-positive values. Adding 1 to all values.")
                data = data + 1
        else:
            if np.any(data <= 0):
                warnings.warn("Data contains non-positive values. Adding 1 to all values.")
                data = data + 1
        
        if base == 'natural':
            return np.log(data)
        elif base == '10':
            return np.log10(data)
        elif base == '2':
            return np.log2(data)
        else:
            raise ValueError("Base must be 'natural', '10', or '2'")
    
    def inverse_transform(self, data: Union[np.ndarray, pd.DataFrame], 
                         method: str) -> Union[np.ndarray, pd.DataFrame]:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data to inverse transform
            method: Normalization method used ('minmax', 'standard', 'robust')
            
        Returns:
            Data in original scale
        """
        if method not in self.fitted_methods:
            raise ValueError(f"Method '{method}' has not been fitted yet.")
        
        if method not in self.scalers:
            raise ValueError(f"Scaler for method '{method}' not found.")
        
        if isinstance(data, pd.DataFrame):
            original = pd.DataFrame(
                self.scalers[method].inverse_transform(data),
                columns=data.columns,
                index=data.index
            )
        else:
            data_reshaped = data.reshape(-1, 1) if data.ndim == 1 else data
            original = self.scalers[method].inverse_transform(data_reshaped)
            if data.ndim == 1:
                original = original.flatten()
                
        return original
    
    def get_scaler_params(self, method: str) -> dict:
        """
        Get parameters of a fitted scaler.
        
        Args:
            method: Normalization method ('minmax', 'standard', 'robust')
            
        Returns:
            Dictionary of scaler parameters
        """
        if method not in self.fitted_methods:
            raise ValueError(f"Method '{method}' has not been fitted yet.")
        
        scaler = self.scalers[method]
        params = {}
        
        if method == 'minmax':
            params['data_min'] = scaler.data_min_
            params['data_max'] = scaler.data_max_
            params['data_range'] = scaler.data_range_
            params['scale'] = scaler.scale_
        elif method == 'standard':
            params['mean'] = scaler.mean_
            params['scale'] = scaler.scale_
            params['var'] = scaler.var_
        elif method == 'robust':
            params['center'] = scaler.center_
            params['scale'] = scaler.scale_
            
        return params
    
    def reset_scalers(self):
        """Reset all fitted scalers."""
        self.scalers.clear()
        self.fitted_methods.clear()


# Utility functions for quick normalization
def quick_minmax(data: Union[np.ndarray, pd.DataFrame], 
                feature_range: Tuple[float, float] = (0, 1)) -> Union[np.ndarray, pd.DataFrame]:
    """Quick Min-Max normalization without storing scaler."""
    handler = NormalizationHandler()
    return handler.min_max_normalize(data, feature_range)

def quick_standardize(data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """Quick standardization without storing scaler."""
    handler = NormalizationHandler()
    return handler.standardize(data)

def quick_robust_scale(data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """Quick robust scaling without storing scaler."""
    handler = NormalizationHandler()
    return handler.robust_scale(data)


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = np.random.randn(100, 3) * 10 + 50
    sample_df = pd.DataFrame(sample_data, columns=['Feature1', 'Feature2', 'Feature3'])
    
    print("=== Normalization Handler Demo ===\n")
    
    # Initialize handler
    normalizer = NormalizationHandler()
    
    # Original data statistics
    print("Original Data Statistics:")
    print(f"Mean: {np.mean(sample_data, axis=0)}")
    print(f"Std: {np.std(sample_data, axis=0)}")
    print(f"Min: {np.min(sample_data, axis=0)}")
    print(f"Max: {np.max(sample_data, axis=0)}\n")
    
    # Min-Max Normalization
    minmax_normalized = normalizer.min_max_normalize(sample_data)
    print("Min-Max Normalized (0-1):")
    print(f"Mean: {np.mean(minmax_normalized, axis=0)}")
    print(f"Min: {np.min(minmax_normalized, axis=0)}")
    print(f"Max: {np.max(minmax_normalized, axis=0)}\n")
    
    # Standardization
    standardized = normalizer.standardize(sample_data)
    print("Standardized (Z-score):")
    print(f"Mean: {np.mean(standardized, axis=0)}")
    print(f"Std: {np.std(standardized, axis=0)}\n")
    
    # Robust Scaling
    robust_scaled = normalizer.robust_scale(sample_data)
    print("Robust Scaled:")
    print(f"Median: {np.median(robust_scaled, axis=0)}")
    print(f"IQR: {np.percentile(robust_scaled, 75, axis=0) - np.percentile(robust_scaled, 25, axis=0)}\n")
    
    # Unit Vector Normalization
    unit_normalized = normalizer.unit_vector_normalize(sample_data)
    print("Unit Vector Normalized (L2):")
    print(f"L2 Norm of first sample: {np.linalg.norm(unit_normalized[0])}\n")
    
    # Custom Range Normalization
    custom_normalized = normalizer.custom_range_normalize(sample_data, -5, 5)
    print("Custom Range Normalized (-5 to 5):")
    print(f"Min: {np.min(custom_normalized, axis=0)}")
    print(f"Max: {np.max(custom_normalized, axis=0)}\n")
    
    # Inverse Transform
    original_recovered = normalizer.inverse_transform(minmax_normalized, 'minmax')
    print("Inverse Transform Test:")
    print(f"Original vs Recovered difference: {np.max(np.abs(sample_data - original_recovered))}\n")
    
    # Scaler Parameters
    print("Min-Max Scaler Parameters:")
    params = normalizer.get_scaler_params('minmax')
    for key, value in params.items():
        print(f"{key}: {value}")
