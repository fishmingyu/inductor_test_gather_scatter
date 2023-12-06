
from torch_geometric.utils import to_scipy_sparse_matrix
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from torch._inductor.select_algorithm import extern_kernels
from torch.utils.cpp_extension import load
from utils import GraphDataset

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor

module = load(
    name='seg_kernel',
    sources=['./openmp/seg_kernel.cpp'],
    extra_cflags=['-O2'],
    verbose=True)


def call_seg_parallel(args):
    indices, values = args
    args.clear()
    out = empty_strided(indices.size(), indices.stride(),
                        device='cpu', dtype=torch.float32)
    module.seg_parallel(indices, values, out)
    del indices
    del values
    return (out, )


def call_seg_atomic(args):
    indices, values = args
    args.clear()
    out = empty_strided(indices.size(), indices.stride(),
                        device='cpu', dtype=torch.float32)
    module.seg_atomic(indices, values, out)
    del indices
    del values
    return (out, )


def call_seg_sequential(args):
    indices, values = args
    args.clear()
    out = empty_strided(indices.size(), indices.stride(),
                        device='cpu', dtype=torch.float32)
    module.seg_sequential(indices, values, out)
    del indices
    del values
    return (out, )


def benchmark_compiled_module_parallel(times=10, repeat=10, indices=None, values=None):
    from torch._inductor.utils import print_performance

    return print_performance(lambda: call_seg_parallel([indices, values]), times=times, repeat=repeat)


def benchmark_compiled_module_atomic(times=10, repeat=10, indices=None, values=None):
    from torch._inductor.utils import print_performance

    return print_performance(lambda: call_seg_atomic([indices, values]), times=times, repeat=repeat)


def benchmark_compiled_module_sequential(times=10, repeat=10, indices=None, values=None):
    from torch._inductor.utils import print_performance

    return print_performance(lambda: call_seg_sequential([indices, values]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    # Add arguments
    num_elements = 10000
    num_reduce = 10
    indices = torch.randint(0, num_reduce, (num_elements, ), dtype=torch.int64)
    values = torch.rand(num_elements, dtype=torch.float32)
    # sort indices
    indices, _ = torch.sort(indices)
    benchmark_compiled_module_parallel(indices=indices, values=values)
    benchmark_compiled_module_atomic(indices=indices, values=values)
    benchmark_compiled_module_sequential(indices=indices, values=values)
