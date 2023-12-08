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
kernel_vec(int feature_size, int edge_num, const long *in_ptr0,
           const float *in_ptr1,
           float *out_ptr0) { // in_ptr0: edge_index, in_ptr1: input_feature,
                              // out_ptr0: output_feature
#pragma omp parallel num_threads(8)
  {
#pragma omp for
    for (long i0 = 0L; i0 < feature_size; i0 += 8L) {
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

extern "C" void
kernel_no_vec_lf(int feature_size, int edge_num, const long *in_ptr0,
                 const float *in_ptr1,
                 float *out_ptr0) { // in_ptr0: edge_index, in_ptr1:
                                    // input_feature, out_ptr0: output_feature
#pragma omp parallel num_threads(8)
  {
#pragma omp for
    for (long i0 = 0L; i0 < static_cast<long>(feature_size); i0 += 1L) {
      float tmp_acc = 0;
      int last_index = -1;
      for (long i1 = 0L; i1 < static_cast<long>(edge_num); i1 += 1L) {
        auto in_node = in_ptr0[static_cast<long>(i1)];
        // Load the current vectorized data
        auto tmp0 = in_ptr1[i0 + (feature_size * in_node)];
        // Load current segment index (assuming scalar index for simplicity)
        auto current_node = in_ptr0[static_cast<long>(edge_num + i1)];
        // Check if the current index is different from the last one
        if (current_node != last_index) {
          // Store the accumulated value to the output (if not the first
          // iteration)
          if (last_index != -1) {
            out_ptr0[last_index * feature_size + i0] = tmp_acc;
          }
          // Reset the accumulator and update the last index
          tmp_acc = tmp0;
          last_index = current_node;
        } else {
          // Same index, accumulate the values
          tmp_acc = tmp_acc + tmp0;
        }
      }
      // Store the last accumulated value
      out_ptr0[(last_index * feature_size) + i0] = tmp_acc;
    }
  }
}

extern "C" void kernel_no_vec_sf(long feature_size, long edges,
                                 const long *edge_index, const float *values,
                                 float *res) {
#pragma omp parallel num_threads(8)
  {
    for (long j = 0; j < feature_size; ++j) {
      float acc = 0.0;      // Accumulator for the current index
      long last_index = -1; // Initialize last_index with the first index
      long in_node = 0;
#pragma omp for
      for (long i = 0; i < edges; ++i) {
        if (i == 0) {
          in_node = edge_index[i];
          acc = values[in_node * feature_size + j];
          last_index = edge_index[i + edges];
        } else if (edge_index[i + edges] != last_index) {
          atomic_add(&res[last_index],
                     acc); // Update the last index with accumulated value
          in_node = edge_index[i];
          acc = values[in_node * feature_size + j];
          last_index = edge_index[i + edges]; // Update the last index
        } else {
          // Same index, accumulate the values
          in_node = edge_index[i];
          acc += values[in_node * feature_size + j];
        }
      }
      // Update the last index after the loop
      atomic_add(&res[last_index], acc);
    }
  }
}

torch::Tensor segment_spmm_vec(torch::Tensor edge_index,
                               torch::Tensor input_feature,
                               torch::Tensor output_feature) {
  auto input_feature_ptr = input_feature.data_ptr<float>();
  auto edge_index_ptr = edge_index.data_ptr<long>();
  auto output_feature_ptr = output_feature.data_ptr<float>();
  auto feature_size = input_feature.size(1);
  auto edge_num = edge_index.size(1);
  kernel_vec(feature_size, edge_num, edge_index_ptr, input_feature_ptr,
             output_feature_ptr);
  return output_feature;
}

torch::Tensor segment_spmm_lf(torch::Tensor edge_index,
                              torch::Tensor input_feature,
                              torch::Tensor output_feature) {
  auto input_feature_ptr = input_feature.data_ptr<float>();
  auto edge_index_ptr = edge_index.data_ptr<long>();
  auto output_feature_ptr = output_feature.data_ptr<float>();
  auto feature_size = input_feature.size(1);
  auto edge_num = edge_index.size(1);
  kernel_no_vec_lf(feature_size, edge_num, edge_index_ptr, input_feature_ptr,
                   output_feature_ptr);
  return output_feature;
}

torch::Tensor segment_spmm_sf(torch::Tensor edge_index,
                              torch::Tensor input_feature,
                              torch::Tensor output_feature) {
  auto input_feature_ptr = input_feature.data_ptr<float>();
  auto edge_index_ptr = edge_index.data_ptr<long>();
  auto output_feature_ptr = output_feature.data_ptr<float>();
  auto feature_size = input_feature.size(1);
  auto edge_num = edge_index.size(1);
  kernel_no_vec_sf(feature_size, edge_num, edge_index_ptr, input_feature_ptr,
                   output_feature_ptr);
  return output_feature;
}

PYBIND11_MODULE(spmm_seg, m) {
  m.def("spmm_lf", &segment_spmm_lf, "spmm_seg_lf (CPU)");
  m.def("spmm_sf", &segment_spmm_sf, "spmm_seg_sf (CPU)");
  m.def("spmm_vec", &segment_spmm_vec, "spmm_seg_vec (CPU)");
}