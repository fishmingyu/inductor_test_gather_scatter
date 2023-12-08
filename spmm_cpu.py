
from torch_geometric.utils import to_scipy_sparse_matrix
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
import time
from math import inf, nan
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from torch._inductor.select_algorithm import extern_kernels
from torch.utils.cpp_extension import load
from utils import GraphDataset

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor

module = load(
    name='spmm',
    sources=['./openmp/spmm.cpp'],
    extra_cflags=['-O2'],
    verbose=True)

module_csr = load(
    name='spmm_csr',
    sources=['./openmp/spmm_csr.cpp'],
    extra_cflags=['-O2'],
    verbose=True)

module_seg = load(
    name='spmm_seg',
    sources=['./openmp/spmm_seg.cpp'],
    extra_cflags=['-O2'],
    verbose=True)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf0 = empty_strided(arg0_1.size(), arg0_1.stride(),
                         device='cpu', dtype=torch.float32)
    module.spmm(arg1_1, arg0_1, buf0)
    del arg0_1
    del arg1_1
    return (buf0, )


def call_csr(args):
    csrptr, csrind, arg0_1 = args
    args.clear()
    buf0 = empty_strided(arg0_1.size(), arg0_1.stride(),
                         device='cpu', dtype=torch.float32)
    module_csr.spmm(csrptr, csrind, arg0_1, buf0)
    del arg0_1
    del csrptr
    del csrind
    return (buf0, )


def call_seg_lf(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf0 = empty_strided(arg0_1.size(), arg0_1.stride(),
                         device='cpu', dtype=torch.float32)
    module_seg.spmm_lf(arg1_1, arg0_1, buf0)
    del arg0_1
    del arg1_1
    return (buf0, )


def call_seg_sf(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf0 = empty_strided(arg0_1.size(), arg0_1.stride(),
                         device='cpu', dtype=torch.float32)
    module_seg.spmm_sf(arg1_1, arg0_1, buf0)
    del arg0_1
    del arg1_1
    return (buf0, )


def call_seg_vec(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf0 = empty_strided(arg0_1.size(), arg0_1.stride(),
                         device='cpu', dtype=torch.float32)
    module_seg.spmm_vec(arg1_1, arg0_1, buf0)
    del arg0_1
    del arg1_1
    return (buf0, )


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(model, example_inputs, times: int = 1) -> float:
    synchronize()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize()
    t1 = time.perf_counter()
    # GC the result after timing
    assert result is not None
    return (t1 - t0) / times


def print_performance(fn, args=(), times=10, repeat=10, baseline=1.0):
    timings = torch.tensor([timed(fn, args, times) for _ in range(repeat)])
    took = torch.median(timings)
    return took


def benchmark_compiled_module(times=10, repeat=10, arg0_1=None, arg1_1=None):
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


def benchmark_compiled_module_csr(times=10, repeat=10, arg0_1=None, arg1_1=None, num_nodes=0):
    scipy_coo = to_scipy_sparse_matrix(arg1_1, num_nodes=num_nodes)
    scipy_csr = scipy_coo.tocsr()
    rowptr = scipy_csr.indptr
    col = scipy_csr.indices
    weight = torch.ones(col.shape, requires_grad=True)

    tcsr = torch.sparse_csr_tensor(rowptr, col, weight, dtype=torch.float)
    csrptr = tcsr.crow_indices().to(torch.int64)
    csrind = tcsr.col_indices().to(torch.int64)
    return print_performance(lambda: call_csr([csrptr, csrind, arg0_1]), times=times, repeat=repeat) / times


def benchmark_compiled_module_seg_lf(times=10, repeat=10, arg0_1=None, arg1_1=None):
    # Step 1: Get the indices that would sort the target nodes
    sorted_indices = arg1_1[1].argsort()

    # Step 2: Reindex the edge_index tensor using the sorted indices
    sorted_edge_index = arg1_1[:, sorted_indices]

    return print_performance(lambda: call_seg_lf([arg0_1, sorted_edge_index]), times=times, repeat=repeat)


def benchmark_compiled_module_seg_sf(times=10, repeat=10, arg0_1=None, arg1_1=None):
    # Step 1: Get the indices that would sort the target nodes
    sorted_indices = arg1_1[1].argsort()

    # Step 2: Reindex the edge_index tensor using the sorted indices
    sorted_edge_index = arg1_1[:, sorted_indices]

    return print_performance(lambda: call_seg_sf([arg0_1, sorted_edge_index]), times=times, repeat=repeat)


def benchmark_compiled_module_seg_vec(times=10, repeat=10, arg0_1=None, arg1_1=None):
    # Step 1: Get the indices that would sort the target nodes
    sorted_indices = arg1_1[1].argsort()

    # Step 2: Reindex the edge_index tensor using the sorted indices
    sorted_edge_index = arg1_1[:, sorted_indices]

    return print_performance(lambda: call_seg_vec([arg0_1, sorted_edge_index]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    import argparse
    parser = argparse.ArgumentParser(description="spmm arg")
    # Add arguments
    parser.add_argument("-f", "--feature", type=int,
                        default=32, help="feature size")
    parser.add_argument("-d", "--dataset", type=str,
                        default='rand', help="dataset")

    # Parse the arguments
    args = parser.parse_args()

    if args.dataset == 'rand':
        num_edges = 200000
        num_nodes = 10000
        feature_size = args.feature
        arg0_1 = rand_strided((num_nodes, feature_size), (feature_size, 1),
                              device='cpu', dtype=torch.float32)
        arg1_1 = torch.randint(num_nodes, (2, num_edges)).to(torch.int64)
    else:
        dataset = GraphDataset(args.dataset, 'cpu')
        num_nodes = dataset.num_nodes
        feature_size = args.feature
        arg0_1 = rand_strided((num_nodes, feature_size), (feature_size, 1),
                              device='cpu', dtype=torch.float32)
        arg1_1 = dataset.edge_index

    atomic_time = benchmark_compiled_module(
        times=20, repeat=10, arg0_1=arg0_1, arg1_1=arg1_1)
    csr_time = benchmark_compiled_module_csr(
        times=20, repeat=10, arg0_1=arg0_1, arg1_1=arg1_1, num_nodes=num_nodes)
    seg_time = benchmark_compiled_module_seg_lf(
        times=20, repeat=10, arg0_1=arg0_1, arg1_1=arg1_1)
    if feature_size >= 32:
        seg_vec_time = benchmark_compiled_module_seg_vec(
            times=20, repeat=10, arg0_1=arg0_1, arg1_1=arg1_1)
        print(f"atomic_time: {atomic_time*1000:.4f} ms \n csr_time: {csr_time*1000:.4f} ms \n seg_time: {seg_time*1000:.4f} ms \n seg_vec_time: {seg_vec_time*1000:.4f} ms")
    else:
        print(
            f"atomic_time: {atomic_time*1000:.4f} ms \n csr_time: {csr_time*1000:.4f} ms \n seg_time: {seg_time*1000:.4f} ms")
