#include "instrinsic.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <omp.h>
#include <torch/extension.h>

#include <ATen/NumericUtils.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>

#include <c10/util/BFloat16-math.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

extern "C" void kernel(int feature_size, int node_num, int edge_num,
                       const long *in_ptr0, const float *in_ptr1,
                       float *out_ptr0) {
#pragma omp parallel num_threads(8)
  {
    {
#pragma omp for
      for (long i0 = static_cast<long>(0L); i0 < static_cast<long>(edge_num);
           i0 += static_cast<long>(1L)) {
#pragma GCC ivdep
        for (long i1 = static_cast<long>(0L);
             i1 < static_cast<long>(feature_size);
             i1 += static_cast<long>(1L)) {
          auto tmp0 = in_ptr0[static_cast<long>(edge_num + i0)];
          auto tmp1 = in_ptr0[static_cast<long>(i0)];
          auto tmp2 = in_ptr1[static_cast<long>(i1 + (feature_size * tmp1))];
          atomic_add(&out_ptr0[static_cast<long>(i1 + (feature_size * tmp0))],
                     tmp2);
        }
      }
    }
  }
}

torch::Tensor spmm(torch::Tensor edge_index, torch::Tensor input_feature,
                   torch::Tensor output_feature) {
  auto input_feature_ptr = input_feature.data_ptr<float>();
  auto edge_index_ptr = edge_index.data_ptr<long>();
  auto output_feature_ptr = output_feature.data_ptr<float>();
  auto feature_size = input_feature.size(1);
  auto node_num = input_feature.size(0);
  auto edge_num = edge_index.size(1);
  kernel(feature_size, node_num, edge_num, edge_index_ptr, input_feature_ptr,
         output_feature_ptr);
  return output_feature;
}

PYBIND11_MODULE(spmm, m) { m.def("spmm", &spmm, "spmm (CPU)"); }