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

extern "C" void kernel(int feature_size, int node_num, const long *csrptr,
                       const long *csrind, const float *in_feature,
                       float *out_feature) {
#pragma omp parallel num_threads(8)
  {
    {
#pragma omp for
      for (long i0 = static_cast<long>(0L); i0 < static_cast<long>(node_num);
           i0 += static_cast<long>(1L)) {
#pragma GCC ivdep
        for (long i1 = static_cast<long>(0L);
             i1 < static_cast<long>(feature_size);
             i1 += static_cast<long>(1L)) {
          auto acc = 0.0;
          for (long id = csrptr[static_cast<long>(i0)];
               id < csrptr[static_cast<long>(i0 + 1L)]; id++) {
            auto tmp0 = csrind[id];
            auto tmp1 =
                in_feature[static_cast<long>(i1 + (feature_size * tmp0))];
            acc += tmp1;
          }
          out_feature[static_cast<long>(i1 + (feature_size * i0))] = acc;
        }
      }
    }
  }
}

torch::Tensor spmm_csr(torch::Tensor csrptr, torch::Tensor csrind,
                       torch::Tensor input_feature,
                       torch::Tensor output_feature) {
  auto csrptr_ptr = csrptr.data_ptr<long>();
  auto csrind_ptr = csrind.data_ptr<long>();
  auto input_feature_ptr = input_feature.data_ptr<float>();
  auto output_feature_ptr = output_feature.data_ptr<float>();
  auto feature_size = input_feature.size(1);
  auto node_num = input_feature.size(0);
  kernel(feature_size, node_num, csrptr_ptr, csrind_ptr, input_feature_ptr,
         output_feature_ptr);
  return output_feature;
}

PYBIND11_MODULE(spmm_csr, m) { m.def("spmm", &spmm_csr, "spmm_csr (CPU)"); }