from sklearn import preprocessing
import numpy as np

# Example data
x_array = np.array([2, 3, 5, 6, 7, 4, 8, 7, 6])
array (L2 with scikit-learn):", normalized_l2[0])

# --- Manual Min-Max Normalization ---
def min_max_norm(dataset):
    min_val = min(dataset)
    max_val = max(dataset)
    return [(value - min_val) / (max_val - min_val) for value in dataset]

normalized_minmax = min_max_norm(x_array)
print("Min-Max normalized:", normalized_minmax)

# --- Standardization ---
scaler = preprocessing.StandardScaler()
normalized_standard = scaler.fit_transform(x_array.reshape(-1, 1))
print("Standardized:", normalized_standard) 