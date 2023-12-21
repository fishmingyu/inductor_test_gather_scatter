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

extern "C" void process_arrays(long n, float *res, const long *indices,
                               const float *values) {

int num_threads = omp_get_num_threads();
#pragma omp parallel num_threads(num_threads)
  {
    float acc = 0.0;      // Accumulator for the current index
    long last_index = -1; // Initialize last_index with the first index

#pragma omp for
    for (long i = 0; i < n; ++i) {
      if (i == 0) {
        acc = values[i];
        last_index = indices[i];
      } else if (indices[i] != last_index) {
        atomic_add(&res[last_index],
                   acc);         // Update the last index with accumulated value
        acc = values[i];         // Reset accumulator for new index
        last_index = indices[i]; // Update the last index
      } else {
        // Same index, accumulate the values
        acc += values[i];
      }
    }
    // Update the last index after the loop
    atomic_add(&res[last_index], acc);
  }
}

extern "C" void process_array_atomic(long n, float *res, const long *indices,
                                     const float *values) {
  // Check if indices and values vectors are empty
  if (n == 0)
    return;
#pragma omp parallel num_threads(8)
  {
#pragma omp for nowait
    for (long i = 0; i < n; ++i) {
      atomic_add(&res[indices[i]], values[i]);
    }
  }
}

extern "C" void process_array_sequential(long n, float *res,
                                         const long *indices,
                                         const float *values) {
  // Check if indices and values vectors are empty
  if (n == 0)
    return;

  float acc = 0.0;              // Accumulator for the current index
  long last_index = indices[0]; // Initialize last_index with the first index

  // Set the first value outside the loop to handle edge cases
  acc = values[0];

  for (long i = 1; i < n; ++i) {
    if (indices[i] != last_index) {
      // Different index encountered, update the last index with accumulated
      // value
      res[last_index] += acc;
      acc = values[i];         // Reset accumulator for new index
      last_index = indices[i]; // Update the last index
    } else {
      // Same index, accumulate the values
      acc += values[i];
    }
  }

  // Update the last index after the loop
  res[last_index] += acc;
}

torch::Tensor seg_parallel(torch::Tensor indices, torch::Tensor values,
                           torch::Tensor out) {
  auto indices_ptr = indices.data_ptr<long>();
  auto values_ptr = values.data_ptr<float>();
  auto out_ptr = out.data_ptr<float>();
  auto n = indices.size(0);
  process_arrays(n, out_ptr, indices_ptr, values_ptr);
  return out;
}

torch::Tensor seg_atomic(torch::Tensor indices, torch::Tensor values,
                         torch::Tensor out) {
  auto indices_ptr = indices.data_ptr<long>();
  auto values_ptr = values.data_ptr<float>();
  auto out_ptr = out.data_ptr<float>();
  auto n = indices.size(0);
  process_array_atomic(n, out_ptr, indices_ptr, values_ptr);
  return out;
}

torch::Tensor seg_sequential(torch::Tensor indices, torch::Tensor values,
                             torch::Tensor out) {
  auto indices_ptr = indices.data_ptr<long>();
  auto values_ptr = values.data_ptr<float>();
  auto out_ptr = out.data_ptr<float>();
  auto n = indices.size(0);
  process_array_sequential(n, out_ptr, indices_ptr, values_ptr);
  return out;
}

PYBIND11_MODULE(seg_kernel, m) {
  m.def("seg_parallel", &seg_parallel, "seg_parallel (CPU)");
  m.def("seg_atomic", &seg_atomic, "seg_atomic (CPU)");
  m.def("seg_sequential", &seg_sequential, "seg_sequential (CPU)");
}