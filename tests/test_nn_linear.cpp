#include "nn.hpp"
#include "losses.hpp"
#include "optim.hpp"
using namespace dl; using namespace dl::optim;

int main(){
  Linear fc(5,3);
  Tensor x = Tensor::randn({10,5});
  Tensor y = Tensor::randn({10,3});
  auto out = fc.forward(x);
  auto loss = mse_loss(out,y,true);
  backward(loss);
  Adam opt(fc.parameters(), 1e-2f);
  opt.step();
  opt.zero_grad();
  return 0;
}