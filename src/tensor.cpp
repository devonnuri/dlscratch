#include "tensor.hpp"
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace dl {

  // ===== Tensor ctor =====
  Tensor::Tensor(std::vector<int64_t> shape_, bool req)
      : storage(numel()), shape(std::move(shape_)), requires_grad(req) {
    // TODO: stride 계산 (row-major)
    if (shape.empty()) return;
    size_t sz = shape.size();
    stride.resize(sz, 1);
    for (size_t i=1; i<sz; i++) {
      stride[sz-i-1] = stride[sz-i] * shape[sz-i];
    }
  }

// ===== static creators =====
  Tensor Tensor::zeros(std::vector<int64_t> shape) {
    Tensor t(shape);
    std::fill(t.data(), t.data() + t.numel(), 0);
    return t;
  }

  Tensor Tensor::ones(std::vector<int64_t> shape) {
    Tensor t(shape);
    std::fill(t.data(), t.data() + t.numel(), 1);
    return t;
  }

  Tensor Tensor::randn(std::vector<int64_t> shape, uint64_t seed) {
    Tensor t(shape);
    std::mt19937 mt(seed);
    std::normal_distribution<float> dist(0.0, 1.0);
    for (size_t i=0; i<t.numel(); i++) {
      *(t.data() + i) = dist(mt);
    }
    return t;
  }

// ===== 기본 메서드 =====
  size_t Tensor::numel() const {
    size_t res=1;
    for (auto x : shape) res *= x;
    return res;
  }

  float* Tensor::data() {
    return storage.data->data() + offset;
  }

  const float* Tensor::data() const {
    return storage.data->data() + offset;
  }

// ===== 뷰 연산 =====
  Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    size_t res=1;
    for (auto x : new_shape) res *= x;
    assert(numel() == res);
    Tensor out(new_shape, requires_grad);
    out.storage = storage;
    return out;
  }

  Tensor Tensor::transpose(int dim0, int dim1) const {
    Tensor out(shape, requires_grad);
    std::swap(out.stride[dim0], out.stride[dim1]);
    std::swap(out.shape[dim0], out.shape[dim1]);

    return out;
  }

  Tensor Tensor::contiguous() const {
    // TODO: stride가 row-major 아니면 새 버퍼에 복사
    bool is_row_major = true;
    size_t sz = shape.size();
    for (size_t i=1; i<sz; i++) {
      if (stride[sz-i] != stride[sz-i+1] * shape[sz-i]) {
        is_row_major = false; break;
      }
    }
    if (is_row_major) return *this;

    Tensor out(shape, requires_grad);
    int i=0, j=0;
    for (size_t k=0;k<numel();k++) {
      for (int s : stride) {

      }
    }
  }

  Tensor Tensor::slice(int dim, int64_t start, int64_t end, int64_t step) const {
    // TODO: slice view 생성
    return Tensor();
  }

// ===== 연산 함수들 =====
  Tensor add(const Tensor& a, const Tensor& b) {
    // TODO: broadcasting 처리 후 elementwise add
    return Tensor();
  }

  Tensor mul(const Tensor& a, const Tensor& b) {
    // TODO: elementwise mul
    return Tensor();
  }

  Tensor matmul(const Tensor& a, const Tensor& b) {
    // TODO: naive O(n^3) 행렬곱
    return Tensor();
  }

  Tensor sum(const Tensor& a, int axis, bool keepdim) {
    // TODO: 특정 axis 합치기
    return Tensor();
  }

  Tensor mean(const Tensor& a, int axis, bool keepdim) {
    // TODO: sum 결과 / 요소수
    return Tensor();
  }

} // namespace dl
