#pragma once
#include "tensor.hpp"
#include <string>
#include <utility>

namespace dl {
  struct Dataset {
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    virtual std::pair<Tensor, Tensor> get(size_t idx) const = 0; // (x,y)
  };

  struct MNIST : public Dataset {
    // 내부에 이미지/라벨 저장
    MNIST(const std::string& root, bool train=true);
    size_t size() const override; // TODO
    std::pair<Tensor, Tensor> get(size_t idx) const override; // TODO
  };

  struct DataLoader {
    const Dataset& ds;
    size_t batch, cur=0;
    bool shuffle;
    std::vector<size_t> index;
    DataLoader(const Dataset& ds, size_t batch=64, bool shuffle=true, uint64_t seed=0);
    bool next(Tensor& x, Tensor& y); // TODO: 배치 반환, 없으면 false
  };
}