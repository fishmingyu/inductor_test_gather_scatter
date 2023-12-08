
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
    name='spmv',
    sources=['./openmp/spmv.cpp'],
    extra_cflags=['-O2'],
    verbose=True)


def call_spmv_parallel(args):
    edge_index, values = args
    args.clear()
    out = empty_strided(values.size(), values.stride(),
                        device='cpu', dtype=torch.float32)
    module.spmv_parallel(edge_index, values, out)
    del edge_index
    del values
    return (out, )


def call_spmv_atomic(args):
    edge_index, values = args
    args.clear()
    out = empty_strided(values.size(), values.stride(),
                        device='cpu', dtype=torch.float32)
    module.spmv_atomic(edge_index, values, out)
    del edge_index
    del values
    return (out, )


def call_spmv_sequential(args):
    edge_index, values = args
    args.clear()
    out = empty_strided(values.size(), values.stride(),
                        device='cpu', dtype=torch.float32)
    module.spmv_sequential(edge_index, values, out)
    del edge_index
    del values
    return (out, )


def benchmark_compiled_module_parallel(times=10, repeat=100, edge_index=None, values=None):
    from torch._inductor.utils import print_performance

    return print_performance(lambda: call_spmv_parallel([edge_index, values]), times=times, repeat=repeat)


def benchmark_compiled_module_atomic(times=10, repeat=100, edge_index=None, values=None):
    from torch._inductor.utils import print_performance

    return print_performance(lambda: call_spmv_atomic([edge_index, values]), times=times, repeat=repeat)


def benchmark_compiled_module_sequential(times=10, repeat=100, edge_index=None, values=None):
    from torch._inductor.utils import print_performance

    return print_performance(lambda: call_spmv_sequential([edge_index, values]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    # Add arguments
    num_elements = 10000
    num_reduce = 100
    edge_index = torch.randint(
        0, num_reduce, (2, num_elements), dtype=torch.int64)
    values = torch.rand(num_elements, dtype=torch.float32)
    # sort edge_index by the second column
    edge_index = edge_index[:, edge_index[1].argsort()]
    benchmark_compiled_module_parallel(edge_index=edge_index, values=values)
    benchmark_compiled_module_atomic(edge_index=edge_index, values=values)
    benchmark_compiled_module_sequential(edge_index=edge_index, values=values)
