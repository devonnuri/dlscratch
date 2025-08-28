#pragma once
#include <vector>
#include <memory>
#include <cassert>
#include <initializer_list>

namespace dl {

  struct Storage {
    std::shared_ptr<std::vector<float>> data;
    Storage(size_t n=0): data(std::make_shared<std::vector<float>>(n)) {}
  };

  struct Tensor {
    Storage storage;           // 공유 스토리지
    size_t offset = 0;         // 시작 오프셋
    std::vector<int64_t> shape;
    std::vector<int64_t> stride; // row-major 기본
    bool requires_grad = false;
    // autograd
    struct Function* grad_fn = nullptr;
    std::shared_ptr<Tensor> grad; // 누적 기울기

    Tensor() = default;
    explicit Tensor(std::vector<int64_t> shape_, bool req=false);

    static Tensor zeros(std::vector<int64_t> shape);
    static Tensor ones(std::vector<int64_t> shape);
    static Tensor randn(std::vector<int64_t> shape, uint64_t seed=0);

    size_t numel() const;
    float* data();
    const float* data() const;

    // view ops
    Tensor reshape(const std::vector<int64_t>& new_shape) const; // TODO: 체크 & 스트라이드 업데이트
    Tensor transpose(int dim0, int dim1) const;                  // TODO
    Tensor contiguous() const;                                   // TODO: 필요 시 복사

    // indexing (간단 버전)
    Tensor slice(int dim, int64_t start, int64_t end, int64_t step=1) const; // TODO
  };

  Tensor add(const Tensor& a, const Tensor& b);
  Tensor mul(const Tensor& a, const Tensor& b);
  Tensor matmul(const Tensor& a, const Tensor& b);
  Tensor sum(const Tensor& a, int axis=-1, bool keepdim=false);
  Tensor mean(const Tensor& a, int axis=-1, bool keepdim=false);

} // namespace dl