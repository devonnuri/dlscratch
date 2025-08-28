#pragma once
#include "tensor.hpp"
#include <functional>
#include <vector>

namespace dl {

  struct Function {
    std::vector<std::shared_ptr<Tensor>> saved_tensors;
    std::vector<Tensor> inputs;
    virtual ~Function() = default;
    virtual void backward(const Tensor& grad_output) = 0; // 각 입력의 grad 누적
  };

  void backward(Tensor& loss); // 토폴로지 역순으로 실행

// 예: AddBackward
  struct AddBackward : public Function {
    Tensor out;
    AddBackward(const Tensor& a, const Tensor& b, const Tensor& out_);
    void backward(const Tensor& grad_output) override; // TODO: 브로드캐스팅 축소
  };

// MulBackward, MatmulBackward, SumBackward 등 선언

} // namespace dl