#pragma once
#include "tensor.hpp"
#include <string>
#include <unordered_map>

namespace dl {

  struct Parameter : public Tensor {
    using Tensor::Tensor;
    Parameter(std::vector<int64_t> shape);
  };

  struct Module {
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& x) = 0;
    virtual std::vector<Parameter*> parameters() { return {}; }
  };

  struct Linear : public Module {
    Parameter W; // [out,in]
    Parameter b; // [out]
    Linear(int64_t in_f, int64_t out_f);
    Tensor forward(const Tensor& x) override; // x:[N,in] → [N,out]
    std::vector<Parameter*> parameters() override { return {&W,&b}; }
  };

  struct Sequential : public Module {
    std::vector<std::shared_ptr<Module>> modules;
    Sequential(std::initializer_list<std::shared_ptr<Module>> ms) : modules(ms) {}
    Tensor forward(const Tensor& x) override; // 순차 적용
    std::vector<Parameter*> parameters() override;
  };

} // namespace dl