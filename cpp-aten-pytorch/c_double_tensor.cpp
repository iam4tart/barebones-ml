// libtorch
#include <torch/extension.h>
#include <vector>

// simple function
torch::Tensor double_tensor(torch::Tensor input) {
    return input*2;
}

// register with pytorch
TORCH_LIBRARY(c_double_tensor, m) {
    m.def("double_tensor", double_tensor);
}