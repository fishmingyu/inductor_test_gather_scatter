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

extern "C" void spmv_parallel_kernel(long edges, float *res,
                                     const long *edge_index,
                                     const float *values) {
#pragma omp parallel
  {
    float acc = 0.0;      // Accumulator for the current index
    long last_index = -1; // Initialize last_index with the first index
    long in_node = 0;

#pragma omp for
    for (long i = 0; i < edges; ++i) {
      if (i == 0) {
        in_node = edge_index[i];
        acc = values[in_node];
        last_index = edge_index[i + edges];
      } else if (edge_index[i + edges] != last_index) {
        atomic_add(&res[last_index],
                   acc); // Update the last index with accumulated value
        in_node = edge_index[i];
        acc = values[in_node];              // Reset accumulator for new index
        last_index = edge_index[i + edges]; // Update the last index
      } else {
        // Same index, accumulate the values
        in_node = edge_index[i];
        acc += values[in_node];
      }
    }
    // Update the last index after the loop
    atomic_add(&res[last_index], acc);
  }
}

extern "C" void spmv_atomic_kernel(long edges, float *res,
                                   const long *edge_index,
                                   const float *values) {
#pragma omp parallel
  {
#pragma omp for
    for (long i = 0; i < edges; ++i) {
      long in_node = edge_index[i];
      long out_node = edge_index[i + edges];
      float val = values[in_node];
      atomic_add(&res[out_node], val);
    }
  }
}

extern "C" void spmv_sequential_kernel(long edges, float *res,
                                       const long *edge_index,
                                       const float *values) {
  float acc = 0.0; // Accumulator for the current index
  long in_node = edge_index[0];
  long last_index = edge_index[0 + edges];

  // Set the first value outside the loop to handle edge cases
  acc = values[in_node];

  for (long i = 1; i < edges; ++i) {
    if (edge_index[i + edges] != last_index) {
      // Different index encountered, update the last index with accumulated
      // value
      res[last_index] += acc;
      in_node = edge_index[i];
      acc = values[in_node];              // Reset accumulator for new index
      last_index = edge_index[i + edges]; // Update the last index
    } else {
      // Same index, accumulate the values
      in_node = edge_index[i];
      acc += values[in_node];
    }
  }

  // Update the last index after the loop
  res[last_index] += acc;
}

torch::Tensor spmv_parallel(torch::Tensor edge_index, torch::Tensor values,
                            torch::Tensor out) {
  auto edge_index_ptr = edge_index.data_ptr<long>();
  auto values_ptr = values.data_ptr<float>();
  auto out_ptr = out.data_ptr<float>();
  auto edges = edge_index.size(1);
  spmv_parallel_kernel(edges, out_ptr, edge_index_ptr, values_ptr);
  return out;
}

torch::Tensor spmv_atomic(torch::Tensor edge_index, torch::Tensor values,
                          torch::Tensor out) {
  auto edge_index_ptr = edge_index.data_ptr<long>();
  auto values_ptr = values.data_ptr<float>();
  auto out_ptr = out.data_ptr<float>();
  auto edges = edge_index.size(1);
  spmv_atomic_kernel(edges, out_ptr, edge_index_ptr, values_ptr);
  return out;
}

torch::Tensor spmv_sequential(torch::Tensor edge_index, torch::Tensor values,
                              torch::Tensor out) {
  auto edge_index_ptr = edge_index.data_ptr<long>();
  auto values_ptr = values.data_ptr<float>();
  auto out_ptr = out.data_ptr<float>();
  auto edges = edge_index.size(1);
  spmv_sequential_kernel(edges, out_ptr, edge_index_ptr, values_ptr);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmv_parallel", &spmv_parallel, "spmv_parallel");
  m.def("spmv_atomic", &spmv_atomic, "spmv_atomic");
  m.def("spmv_sequential", &spmv_sequential, "spmv_sequential");
}