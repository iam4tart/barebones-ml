import torch
import c_double_tensor

x = torch.Tensor([1.0, 2.0, 3.0])
y = c_double_tensor.double_tensor(x)

print("Input: ", x)
print("Output: ", y)