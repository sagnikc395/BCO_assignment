import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# tensors can be created from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# new tensor retains the properties (shape,datatype) of the argument tensor, unless explicitly overidden
#
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"ones tensor: {x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype
print(f"random tensor: {x_rand}\n")


shape = (
    2,
    3,
)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"random tensor: \n{rand_tensor}")
print(f"ones tensor: \n{ones_tensor}")
print(f"zeros tensor: \n{zeros_tensor}")

tensor = torch.rand(3, 4)
print(f"shape of tensor: {tensor.shape}")
print(f"datatype of tensor: {tensor.dtype}")
print(f"device tensor is stored on: {tensor.device}")
