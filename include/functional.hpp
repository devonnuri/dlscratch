#pragma once
#include "tensor.hpp"

namespace dl {
  Tensor relu(const Tensor& x);
  Tensor sigmoid(const Tensor& x);
  Tensor tanh_(const Tensor& x);
  Tensor softmax(const Tensor& x, int axis=-1);
}
