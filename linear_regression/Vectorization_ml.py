import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

def demonstrate_vectorization():
    """
    Comprehensive demonstration of vectorization in ML operations
    """
    print("=== Vectorization in Machine Learning Demo ===\n")
    
    # Generate sample data
    X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42)
    X = StandardScaler().fit_transform(X)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}\n")
    
    # 1. Basic Operations: Element-wise multiplication
    print("1. ELEMENT-WISE OPERATIONS")
    print("-" * 40)
    
    # Non-vectorized version
    def multiply_loops(X, weights):
        result = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                result[i, j] = X[i, j] * weights[j]
        return result
    
    # Vectorized version
    def multiply_vectorized(X, weights):
        return X * weights
    
    weights = np.random.randn(X.shape[1])
    
    # Time comparison
    start = time.time()
    result_loops = multiply_loops(X[:1000], weights)  # Use subset for loops
    time_loops = time.time() - start
    
    start = time.time()
    result_vectorized = multiply_vectorized(X, weights)
    time_vectorized = time.time() - start
    
    print(f"Loop version (1000 samples): {time_loops:.4f} seconds")
    print(f"Vectorized version (10000 samples): {time_vectorized:.4f} seconds")
    print(f"Speedup: {time_loops/time_vectorized:.1f}x faster\n")
    
    # 2. Linear Regression Forward Pass
    print("2. LINEAR REGRESSION FORWARD PASS")
    print("-" * 40)
    
    # Non-vectorized version
    def linear_forward_loops(X, weights, bias):
        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            pred = bias
            for j in range(X.shape[1]):
                pred += X[i, j] * weights[j]
            predictions[i] = pred
        return predictions
    
    # Vectorized version
    def linear_forward_vectorized(X, weights, bias):
        return X @ weights + bias
    
    weights = np.random.randn(X.shape[1])
    bias = np.random.randn()
    
    start = time.time()
    pred_loops = linear_forward_loops(X[:1000], weights, bias)
    time_loops = time.time() - start
    
    start = time.time()
    pred_vectorized = linear_forward_vectorized(X, weights, bias)
    time_vectorized = time.time() - start
    
    print(f"Loop version (1000 samples): {time_loops:.4f} seconds")
    print(f"Vectorized version (10000 samples): {time_vectorized:.4f} seconds")
    print(f"Speedup: {time_loops/time_vectorized:.1f}x faster\n")
    
    # 3. Sigmoid Activation Function
    print("3. SIGMOID ACTIVATION FUNCTION")
    print("-" * 40)
    
    # Non-vectorized version
    def sigmoid_loops(z):
        result = np.zeros_like(z)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]) if len(z.shape) > 1 else range(1):
                if len(z.shape) > 1:
                    result[i, j] = 1 / (1 + np.exp(-z[i, j]))
                else:
                    result[i] = 1 / (1 + np.exp(-z[i]))
        return result
    
    # Vectorized version
    def sigmoid_vectorized(z):
        return 1 / (1 + np.exp(-z))
    
    z = np.random.randn(5000, 10)
    
    start = time.time()
    sig_loops = sigmoid_loops(z[:500])  # Subset for loops
    time_loops = time.time() - start
    
    start = time.time()
    sig_vectorized = sigmoid_vectorized(z)
    time_vectorized = time.time() - start
    
    print(f"Loop version (500×10): {time_loops:.4f} seconds")
    print(f"Vectorized version (5000×10): {time_vectorized:.4f} seconds")
    print(f"Speedup: {time_loops/time_vectorized:.1f}x faster\n")
    
    # 4. Gradient Computation
    print("4. GRADIENT COMPUTATION")
    print("-" * 40)
    
    # Non-vectorized gradient computation
    def compute_gradient_loops(X, y, predictions):
        m = X.shape[0]
        gradient = np.zeros(X.shape[1])
        
        for j in range(X.shape[1]):
            grad_sum = 0
            for i in range(m):
                grad_sum += (predictions[i] - y[i]) * X[i, j]
            gradient[j] = grad_sum / m
        return gradient
    
    # Vectorized gradient computation
    def compute_gradient_vectorized(X, y, predictions):
        m = X.shape[0]
        return X.T @ (predictions - y) / m
    
    predictions = pred_vectorized[:len(y)]
    
    start = time.time()
    grad_loops = compute_gradient_loops(X[:1000], y[:1000], predictions[:1000])
    time_loops = time.time() - start
    
    start = time.time()
    grad_vectorized = compute_gradient_vectorized(X, y, predictions)
    time_vectorized = time.time() - start
    
    print(f"Loop version (1000 samples): {time_loops:.4f} seconds")
    print(f"Vectorized version (10000 samples): {time_vectorized:.4f} seconds")
    print(f"Speedup: {time_loops/time_vectorized:.1f}x faster\n")
    
    # 5. Distance Computation (useful for KNN, clustering)
    print("5. PAIRWISE DISTANCE COMPUTATION")
    print("-" * 40)
    
    # Non-vectorized Euclidean distance
    def euclidean_distance_loops(X1, X2):
        distances = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                dist = 0
                for k in range(X1.shape[1]):
                    dist += (X1[i, k] - X2[j, k]) ** 2
                distances[i, j] = np.sqrt(dist)
        return distances
    
    # Vectorized distance computation
    def euclidean_distance_vectorized(X1, X2):
        # Using broadcasting: ||a-b||² = ||a||² + ||b||² - 2a·b
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1)
        cross_term = 2 * X1 @ X2.T
        distances_sq = X1_sq + X2_sq - cross_term
        return np.sqrt(np.maximum(distances_sq, 0))  # Ensure non-negative
    
    X_subset1 = X[:100]
    X_subset2 = X[100:200]
    
    start = time.time()
    dist_loops = euclidean_distance_loops(X_subset1[:20], X_subset2[:20])
    time_loops = time.time() - start
    
    start = time.time()
    dist_vectorized = euclidean_distance_vectorized(X_subset1, X_subset2)
    time_vectorized = time.time() - start
    
    print(f"Loop version (20×20): {time_loops:.4f} seconds")
    print(f"Vectorized version (100×100): {time_vectorized:.4f} seconds")
    print(f"Speedup: {time_loops/time_vectorized:.1f}x faster\n")
    
    # 6. Mini-batch Processing
    print("6. MINI-BATCH PROCESSING")
    print("-" * 40)
    
    def process_batches_loops(X, batch_size, weights):
        """Process data in mini-batches using loops"""
        n_samples = X.shape[0]
        results = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = X[start_idx:end_idx]
            
            # Process each sample in batch
            batch_result = np.zeros(batch.shape[0])
            for i in range(batch.shape[0]):
                result = 0
                for j in range(batch.shape[1]):
                    result += batch[i, j] * weights[j]
                batch_result[i] = result
            results.append(batch_result)
        # Concatenate results from all batches
        return np.concatenate(results)
    
    def process_batches_vectorized(X, batch_size, weights):
        """Process data in mini-batches using vectorization"""
        n_samples = X.shape[0]
        results = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = X[start_idx:end_idx]
            batch_result = batch @ weights  # Vectorized computation
            results.append(batch_result)
        
        return np.concatenate(results)
    
    batch_size = 256
    weights = np.random.randn(X.shape[1])
    
    start = time.time()
    result_loops = process_batches_loops(X[:2000], batch_size, weights)
    time_loops = time.time() - start
    
    start = time.time()
    result_vectorized = process_batches_vectorized(X, batch_size, weights)
    time_vectorized = time.time() - start
    
    print(f"Loop version (2000 samples): {time_loops:.4f} seconds")
    print(f"Vectorized version (10000 samples): {time_vectorized:.4f} seconds")
    print(f"Speedup: {time_loops/time_vectorized:.1f}x faster\n")
    
    # 7. Advanced: Broadcasting Example
    print("7. BROADCASTING EXAMPLE")
    print("-" * 40)
    
    # Normalize features by subtracting mean and dividing by std
    def normalize_loops(X):
        normalized = np.zeros_like(X)
        for j in range(X.shape[1]):  # For each feature
            mean_j = np.sum(X[:, j]) / X.shape[0]
            var_j = np.sum((X[:, j] - mean_j)**2) / X.shape[0]
            std_j = np.sqrt(var_j)
            
            for i in range(X.shape[0]):  # For each sample
                normalized[i, j] = (X[i, j] - mean_j) / std_j
        return normalized
    
    def normalize_vectorized(X):
        means = np.mean(X, axis=0)  # Shape: (n_features,)
        stds = np.std(X, axis=0)    # Shape: (n_features,)
        return (X - means) / stds   # Broadcasting!
    
    start = time.time()
    norm_loops = normalize_loops(X[:1000])
    time_loops = time.time() - start
    
    start = time.time()
    norm_vectorized = normalize_vectorized(X)
    time_vectorized = time.time() - start
    
    print(f"Loop version (1000 samples): {time_loops:.4f} seconds")
    print(f"Vectorized version (10000 samples): {time_vectorized:.4f} seconds")
    print(f"Speedup: {time_loops/time_vectorized:.1f}x faster\n")
    
    print("=== KEY TAKEAWAYS ===")
    print("1. Vectorization leverages optimized C/Fortran libraries (BLAS, LAPACK)")
    print("2. Eliminates Python loops, reducing interpretation overhead")
    print("3. Enables SIMD (Single Instruction, Multiple Data) operations")
    print("4. Better memory access patterns and cache utilization")
    print("5. Essential for scaling ML algorithms to large datasets")
    print("6. NumPy broadcasting allows element-wise operations on different shapes")

if __name__ == "__main__":
    demonstrate_vectorization()