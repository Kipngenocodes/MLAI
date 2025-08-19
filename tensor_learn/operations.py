# Standard numpy-like indexing and slicing:
import  torch


tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# Joining Tensors can use torch.cat to concatenate a sequence of tensors along a given dimension.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Arithmetic Operations in Tensor 
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)



# single-element tensor
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# Inplace operations Operations that store the result into
# the operand are called in-place. They are denoted by a _ suffix.
tensor.add_(5)
print(tensor)


# Bridge with NumPy
# Tensors on the CPU and NumPy arrays can share their underlying memory locations,
# and changing one will change the other.
# ensor to Numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

