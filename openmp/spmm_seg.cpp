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

extern "C" void
kernel(int feature_size, int node_num, int edge_num, const long *in_ptr0,
       const float *in_ptr1,
       float *out_ptr0) { // in_ptr0: edge_index, in_ptr1: input_feature,
                          // out_ptr0: output_feature
#pragma omp parallel num_threads(8)
  {
#pragma omp for
    for (long i0 = 0L; i0 < feature_size; i0 += 8L) {
      float tmp_acc = 0;
      int last_index = -1;
      at::vec::Vectorized<float> tmp_acc_vec = at::vec::Vectorized<float>(0);
      for (long i1 = 0L; i1 < static_cast<long>(edge_num); i1 += 1L) {
        auto in_node = in_ptr0[static_cast<long>(i1)];
        // Load the current vectorized data
        auto tmp0 =
            at::vec::Vectorized<float>::loadu(in_ptr1 + i0 + (32L * in_node));
        // Load current segment index (assuming scalar index for simplicity)
        auto current_node = in_ptr0[static_cast<long>(edge_num + i1)];
        // Check if the current index is different from the last one
        if (current_node != last_index) {
          // Store the accumulated value to the output (if not the first
          // iteration)
          if (last_index != -1) {
            tmp_acc_vec.store(out_ptr0 + (last_index * 32L) + i0);
          }
          // Reset the accumulator and update the last index
          tmp_acc_vec = tmp0;
          last_index = current_node;
        } else {
          // Same index, accumulate the values
          tmp_acc_vec = tmp_acc_vec + tmp0;
        }
      }
      // Store the last accumulated value
      tmp_acc_vec.store(out_ptr0 + +(last_index * 32L) + i0);
    }
  }
}

torch::Tensor segment_spmm(torch::Tensor edge_index,
                           torch::Tensor input_feature,
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

PYBIND11_MODULE(spmm_seg, m) {
  m.def("spmm", &segment_spmm, "segment spmm (CPU)");
}