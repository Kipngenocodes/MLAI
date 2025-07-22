import numpy as np
import time

# Generate sample data: 1000 points in 3D space
np.random.seed(42)
X = np.random.rand(1000, 3)  # 1000 points with 3 features
Y = np.random.rand(1000, 3)  # Another set of 1000 points

# Unvectorized approach: Using loops to compute Euclidean distances
def unvectorized_euclidean_distance(X, Y):
    distances = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            sum_squares = 0
            for k in range(X.shape[1]):
                sum_squares += (X[i, k] - Y[j, k]) ** 2
            distances[i, j] = np.sqrt(sum_squares)
    return distances

# Vectorized approach: Using NumPy operations
def vectorized_euclidean_distance(X, Y):
    # Compute squared differences: (X - Y)^2
    # X[:, np.newaxis] adds a dimension to X for broadcasting
    squared_diff = (X[:, np.newaxis] - Y) ** 2
    # Sum along feature axis and take square root
    distances = np.sqrt(np.sum(squared_diff, axis=2))
    return distances

# Time the unvectorized approach
start_time = time.time()
unvectorized_result = unvectorized_euclidean_distance(X, Y)
unvectorized_time = time.time() - start_time

# Time the vectorized approach
start_time = time.time()
vectorized_result = vectorized_euclidean_distance(X, Y)
vectorized_time = time.time() - start_time

# Verify results are equivalent
np.testing.assert_almost_equal(unvectorized_result, vectorized_result, decimal=10)

# Print timing results
print(f"Unvectorized time: {unvectorized_time:.4f} seconds")
print(f"Vectorized time: {vectorized_time:.4f} seconds")
print(f"Speedup: {unvectorized_time / vectorized_time:.2f}x")

# Example output for a small subset
print("\nSample distances (first 3x3 block):")
print(vectorized_result[:3, :3])