
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "ATen/ops/mean.h"
#include "ATen/ops/std.h"
#include <ATen/ATen.h>
#pragma GCC diagnostic pop
#include <iostream>

using namespace std;

// libtorch Playground

int main() {
  at::Tensor a = at::ones({2, 2}, at::kInt);
  at::Tensor b = at::randn({4, 3});
  auto c = b * 10;
  cout << a << endl;

  cout << b << endl;
  auto b_min = at::min(b);
  auto b_max = at::max(b);
  cout << 2 * ((b - b_min) / (b_max - b_min)) - 1 << endl;

  cout << c << endl;
  auto c_min = at::min(c);
  auto c_max = at::max(c);
  cout << 2 * ((c - c_min) / (c_max - c_min)) - 1 << endl;
  return 0;
}