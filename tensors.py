"""
source: https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
"""

# Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

# Tensors are similar to NumPy’s ndarrays, except that tensors can **run on GPUs** or other hardware accelerators. In fact, tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data (see Bridge with NumPy). Tensors are also optimized for **automatic differentiation** (we’ll see more about that later in the Autograd section). If you’re familiar with ndarrays, you’ll be right at home with the Tensor API. If not, follow along!

import torch
import numpy as np

print("== Initializing a Tensor ==")

# directly from python data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(f"x_data:\n", x_data)

# from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"x_np:\n", x_np)

# from another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# with random/constant values
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


print("== Attributes of a Tensor ==")

# Tensor attributes describe their 
# - shape,
# - datatype,
# - and the device on which they are stored.

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


print("== Operations on Tensors")

# Over 100 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing), sampling and more are comprehensively described here.

# Each of these operations can be run on the GPU (at typically higher speeds than on a CPU). If you’re using Colab, allocate a GPU by going to Runtime > Change runtime type > GPU.

# By default, tensors are created on the CPU. We need to explicitly move tensors to the GPU using .to method (after checking for GPU availability). Keep in mind that copying large tensors across devices can be expensive in terms of time and memory!

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# Standard numpy-like indexing and slicing:
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)


# Joining tensors 
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Arithmetic operations

# tensor = torch.ones(4)
tensor = torch.tensor([1., 2., 3., 4.])
print(tensor)
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(f"dot product (aka matrix multiplication), y3:\n{y3}")


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(f"element-wise product, z3:\n{z3}")


# Single-element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)


print("Bridge with NumPy")

# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.

print("= Tensor to NumPy array =")
t = torch.ones(5)
n = t.numpy()
print(f"t: {t}")
print(f"n: {n}")

# A change in the tensor reflects in the NumPy array.
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

print("= NumPy array to Tensor =")
n = np.ones(5)
t = torch.from_numpy(n)
print(f"t: {t}")
print(f"n: {n}")

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
