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
  }

// ===== static creators =====
  Tensor Tensor::zeros(std::vector<int64_t> shape) {
    Tensor t(shape);
    // TODO: 데이터 0으로 채우기
    return t;
  }

  Tensor Tensor::ones(std::vector<int64_t> shape) {
    Tensor t(shape);
    // TODO: 데이터 1로 채우기
    return t;
  }

  Tensor Tensor::randn(std::vector<int64_t> shape, uint64_t seed) {
    Tensor t(shape);
    // TODO: 정규분포 난수 채우기
    return t;
  }

// ===== 기본 메서드 =====
  size_t Tensor::numel() const {
    // TODO: shape의 모든 원소 곱하기
    return 0;
  }

  float* Tensor::data() {
    // TODO: storage.data->data() + offset 리턴
    return nullptr;
  }

  const float* Tensor::data() const {
    // TODO: storage.data->data() + offset 리턴
    return nullptr;
  }

// ===== 뷰 연산 =====
  Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    // TODO: 요소수 동일 체크 & stride 새로 계산
    return Tensor();
  }

  Tensor Tensor::transpose(int dim0, int dim1) const {
    // TODO: stride/shape 바꿔서 새 view 만들기
    return Tensor();
  }

  Tensor Tensor::contiguous() const {
    // TODO: stride가 row-major 아니면 새 버퍼에 복사
    return *this;
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
