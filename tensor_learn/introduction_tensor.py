import torch
import numpy as np

# initializing tensor directly from data
data = [[1,2],[3,4]]
tensor = torch.tensor(data)

# initializing tensor directly from numpy array
numpy_array = np.array(data)
tensor_from_numpy = torch.tensor(numpy_array)