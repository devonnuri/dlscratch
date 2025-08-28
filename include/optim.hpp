#pragma once
#include "nn.hpp"

namespace dl { namespace optim {
  struct Optimizer {
    std::vector<Parameter*> params;
    explicit Optimizer(const std::vector<Parameter*>& ps) : params(ps) {}
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad();
  };

  struct SGD : public Optimizer {
    float lr, momentum;
    std::unordered_map<Parameter*, Tensor> v;
    SGD(const std::vector<Parameter*>& ps, float lr, float momentum=0.f);
    void step() override; // TODO
  };

  struct Adam : public Optimizer {
    float lr, beta1, beta2, eps;
    std::unordered_map<Parameter*, Tensor> m, v;
    uint64_t t = 0;
    Adam(const std::vector<Parameter*>& ps, float lr=1e-3f, float b1=0.9f, float b2=0.999f, float eps=1e-8f);
    void step() override; // TODO
  };

}} // namespace dl::optim