#pragma once
#include "tensor.hpp"

namespace dl {
  Tensor mse_loss(const Tensor& pred, const Tensor& target, bool reduction_mean=true);
  Tensor cross_entropy(const Tensor& logits, const Tensor& targets); // targets: class index
}