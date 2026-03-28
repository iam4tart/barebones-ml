import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x * y + y**2

def print_graph(var, prefix=""):
    if var.grad_fn is not None:
        print(prefix + str(var.grad_fn))
        for next_fn, _ in var.grad_fn.next_functions:
            if next_fn is not None:
                print_graph(next_fn, prefix + "  ")

print("Computational Graph for z:")
print_graph(z)

# <AddBackward0 object at 0x...>
#   <PowBackward0 object at 0x...>
#   <MulBackward0 object at 0x...>
#     <AccumulateGrad object at 0x...>
#     <AccumulateGrad object at 0x...>