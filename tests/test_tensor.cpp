#include "tensor.hpp"
#include <cassert>
using namespace dl;

int main(){
  auto a = Tensor::ones({2,3});
  auto b = Tensor::ones({1,3});
  auto c = add(a,b);
  assert(c.shape==std::vector<int64_t>({2,3}));
  // 값 확인 TODO
  auto d = matmul(Tensor::ones({2,4}), Tensor::ones({4,3}));
  assert(d.shape==std::vector<int64_t>({2,3}));
  return 0;
}
