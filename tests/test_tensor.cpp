#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>

using namespace dl;

// ----------------- Helpers -----------------
static bool almost_eq(float a, float b, float eps=1e-5f){
  return std::fabs(a-b) <= eps * (1.0f + std::max(std::fabs(a), std::fabs(b)));
}

static void expect(bool cond, const char* msg){
  if(!cond){ std::cerr << "[FAIL] " << msg << "\n"; std::abort(); }
}

static size_t prod(const std::vector<int64_t>& v){
  size_t p=1; for(auto x: v) p*=static_cast<size_t>(x); return p;
}

static void expect_shape(const Tensor& t, std::vector<int64_t> s){
  expect(t.shape==s, "shape mismatch");
}

static void expect_stride(const Tensor& t, std::vector<int64_t> st){
  expect(t.stride==st, "stride mismatch");
}

static void fill_sequential(Tensor& t, float start=0.f, float step=1.f){
  float val=start;
  auto* p = t.data();
  for(size_t i=0;i<t.numel();++i){ p[i]=val; val+=step; }
}

static float get_at_rowmajor(const Tensor& t, const std::vector<int64_t>& idx){
  size_t off = t.offset;
  for(size_t d=0; d<idx.size(); ++d) off += (size_t)idx[d] * (size_t)t.stride[d];
  return t.data()[off];
}

// Make sure a non-contiguous view becomes contiguous and preserves values
static void expect_contiguous_equal(const Tensor& v, const Tensor& c){
  expect(v.shape==c.shape, "contiguous(): shape altered");
  // brute-force compare via multi-index
  std::vector<int64_t> idx(v.shape.size(),0);
  for(size_t lin=0; lin<c.numel(); ++lin){
    // compute linear in contiguous target
    float vval = get_at_rowmajor(v, idx);
    float cval = c.data()[lin];
    if(!almost_eq(vval, cval)){
      std::cerr << "[FAIL] contiguous value mismatch at lin=" << lin << "\n";
      std::abort();
    }
    // increment multi-index (row-major)
    for(int d=(int)idx.size()-1; d>=0; --d){
      idx[d]++;
      if(idx[d] < v.shape[d]) break;
      idx[d]=0;
    }
  }
}

int main(){
  std::cout << "[RUN] test_tensor.cpp\n";

  // ---- 1) Construction & numel ----
  {
    Tensor a = Tensor::zeros({2,3,4});
    expect_shape(a, {2,3,4});
    expect(a.numel()==24, "numel incorrect");
    // row-major stride expected: (3*4, 4, 1) = (12, 4, 1)
    expect_stride(a, {12,4,1});
    for(size_t i=0;i<a.numel();++i) expect(almost_eq(a.data()[i], 0.f), "zeros not filled");
  }

  // ---- 2) ones/randn fill ----
  {
    Tensor b = Tensor::ones({5});
    expect_shape(b, {5});
    for(size_t i=0;i<b.numel();++i) expect(almost_eq(b.data()[i], 1.f), "ones not filled");

    Tensor c = Tensor::randn({8}, 42);
    expect_shape(c, {8});
    // Only check that not all zeros (weak check)
    float sumv=0.f; for(size_t i=0;i<c.numel();++i) sumv+=c.data()[i];
    expect(std::fabs(sumv) > 1e-6f, "randn looks all zeros");
  }

  // ---- 3) reshape (view) ----
  {
    Tensor a = Tensor::ones({2,3,4}); // contiguous
    Tensor r = a.reshape({3,8});
    expect_shape(r, {3,8});
    // reshape should be a view; storage must be shared
    expect(a.storage.data == r.storage.data, "reshape should share storage");
    // values line-up when read contiguously
    // fill with sequential values to check precise mapping
    fill_sequential(a);
    // after reshape, contiguous() must preserve order
    Tensor rc = r.contiguous();
    for(size_t i=0;i<rc.numel();++i) expect(almost_eq(rc.data()[i], (float)i), "reshape mapping wrong");
  }

  // ---- 4) transpose (view) ----
  {
    Tensor a = Tensor::zeros({2,3});
    fill_sequential(a, 10.f, 1.f); // data = 10,11,12,13,14,15 row-major
    // transpose(0,1): shape (3,2), stride swap (1,3)
    Tensor t = a.transpose(0,1);
    expect_shape(t, {3,2});
    // transpose shares storage
    expect(a.storage.data == t.storage.data, "transpose should share storage");
    // Check a^T values after making contiguous
    Tensor tc = t.contiguous();
    // Expected matrix:
    // a = [[10,11,12],[13,14,15]]  -> a^T = [[10,13],[11,14],[12,15]]
    float expect_vals[6] = {10,13,11,14,12,15};
    for(size_t i=0;i<6;++i) expect(almost_eq(tc.data()[i], expect_vals[i]), "transpose values wrong");
  }

  // ---- 5) slice (view) ----
  {
    // a: shape (3,4) values 0..11
    Tensor a = Tensor::zeros({3,4});
    fill_sequential(a, 0.f, 1.f);
    // slice row 1: shape (4)
    Tensor row1 = a.slice(0, 1, 2); // [1:2) → one row
    expect_shape(row1, {4});
    Tensor row1c = row1.contiguous();
    float exp_r1[4] = {4,5,6,7};
    for(int i=0;i<4;++i) expect(almost_eq(row1c.data()[i], exp_r1[i]), "slice row wrong");
    // slice columns 1..3 of row 2 (zero-based)
    Tensor row2 = a.slice(0, 2, 3); // pick row index 2
    Tensor row2c = row2.slice(0, 0, 1).transpose(0,0).contiguous(); // ensure single row contiguous
    // columns slice
    Tensor sub = a.slice(1, 1, 3); // slice cols [1,3) across all rows → shape (3,2)
    Tensor subc = sub.contiguous();
    // expected [[1,2],[5,6],[9,10]]
    float exp_sub[6] = {1,2,5,6,9,10};
    for(int i=0;i<6;++i) expect(almost_eq(subc.data()[i], exp_sub[i]), "slice cols wrong");
  }

  // ---- 6) add with broadcasting ----
  {
    Tensor A = Tensor::zeros({2,3}); fill_sequential(A, 1.f, 1.f); // [[1,2,3],[4,5,6]]
    Tensor B = Tensor::ones({1,3}); // broadcast over first dim
    Tensor C = add(A, B);
    expect_shape(C, {2,3});
    float exp[6] = {2,3,4,5,6,7};
    for(int i=0;i<6;++i) expect(almost_eq(C.data()[i], exp[i]), "add broadcast wrong");
  }

  // ---- 7) mul with broadcasting (scalar) ----
  {
    Tensor A = Tensor::zeros({2,2}); fill_sequential(A, 2.f, 2.f); // [2,4,6,8]
    Tensor s = Tensor::ones({1}); // scalar-like
    Tensor C = mul(A, s);
    float exp[4] = {2,4,6,8};
    for(int i=0;i<4;++i) expect(almost_eq(C.data()[i], exp[i]), "mul broadcast wrong");
  }

  // ---- 8) matmul ----
  {
    Tensor X = Tensor::zeros({2,3});
    Tensor W = Tensor::zeros({3,2});
    // X = [[1,2,3],[4,5,6]], W = [[1,0],[0,1],[1,1]]
    float xv[6]={1,2,3,4,5,6}; for(int i=0;i<6;++i) X.data()[i]=xv[i];
    float wv[6]={1,0,0,1,1,1}; for(int i=0;i<6;++i) W.data()[i]=wv[i];
    Tensor Y = matmul(X, W);
    expect_shape(Y, {2,2});
    // Y = X*W = [[1+0+3, 0+2+3],[4+0+6,0+5+6]] = [[4,5],[10,11]]
    float exp[4]={4,5,10,11};
    for(int i=0;i<4;++i) expect(almost_eq(Y.data()[i], exp[i]), "matmul wrong");
  }

  // ---- 9) sum / mean ----
  {
    Tensor A = Tensor::zeros({2,3}); fill_sequential(A, 1.f, 1.f); // [[1,2,3],[4,5,6]] sum=21
    Tensor s_all = sum(A, -1, true); // by spec: -1 means sum over all? keepdim=true
    expect(s_all.numel()==1, "sum all should be scalar");
    expect(almost_eq(s_all.data()[0], 21.f), "sum all wrong");

    Tensor s_axis1 = sum(A, 1, false); // sum over columns → shape (2)
    expect_shape(s_axis1, {2});
    expect(almost_eq(s_axis1.data()[0], 6.f), "sum axis1[0] wrong");
    expect(almost_eq(s_axis1.data()[1], 15.f), "sum axis1[1] wrong");

    Tensor m_axis0 = mean(A, 0, false); // mean over rows → shape (3)
    expect_shape(m_axis0, {3});
    float expm[3] = { (1+4)/2.f, (2+5)/2.f, (3+6)/2.f };
    for(int i=0;i<3;++i) expect(almost_eq(m_axis0.data()[i], expm[i]), "mean axis0 wrong");
  }

  // ---- 10) non-contiguous + op sanity (manual contiguous) ----
  {
    Tensor A = Tensor::zeros({2,3}); fill_sequential(A, 1.f, 1.f);
    Tensor T = A.transpose(0,1);           // non-contiguous view (3,2)
    Tensor Tc = T.contiguous();            // make contiguous
    Tensor O = add(Tc, Tensor::ones({1,2}));
    expect_shape(O, {3,2});
    // Expected: (A^T) + 1
    // A^T = [[1,4],[2,5],[3,6]] → +1
    float exp[6] = {2,5,3,6,4,7};
    for(int i=0;i<6;++i) expect(almost_eq(O.data()[i], exp[i]), "op after contiguous wrong");
  }

  std::cout << "[PASS] all tensor tests passed.\n";
  return 0;
}
