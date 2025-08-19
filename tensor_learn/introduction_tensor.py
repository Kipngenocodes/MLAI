import torch
import numpy as np

# initializing tensor directly from data
data = [[1,2],[3,4]]
tensor = torch.tensor(data)
# output: tensor([[1, 2],
#                 [3, 4]])
print(tensor)


# initializing tensor directly from numpy array
numpy_array = np.array(data)
tensor_from_numpy = torch.tensor(numpy_array)

# output: tensor([[1, 2],
#                 [3, 4]])
print(tensor_from_numpy)

# From another tensor (new tensor will have same shape and datatype)
x_ones = torch.ones_like(tensor)

# from another tensor with explicit datatype
x_rand = torch.rand_like(tensor, dtype=torch.float)

# outputs
print("Random Tensor:", x_rand)
print("Ones Tensor:", x_ones)

# shape is a tuple in tensor dimension, It determines the dimensionality of the output tensor.
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# Attribute of tensor 
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")~
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())