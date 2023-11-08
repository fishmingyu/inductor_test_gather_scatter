
import argparse
from torch_geometric.utils import from_scipy_sparse_matrix
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.utils import maybe_profile
import scipy.sparse as sp

from torch import empty_strided, device
from torch._inductor.select_algorithm import extern_kernels
from torch.utils.cpp_extension import load
from torch._dynamo.testing import rand_strided


module = load(
    name='spmm',
    sources=['../openmp/spmm.cpp'],
    extra_cflags=['-O2'],
    verbose=True)

module_csr = load(
    name='spmm_csr',
    sources=['../openmp/spmm_csr.cpp'],
    extra_cflags=['-O2'],
    verbose=True)


def golden_spmm(sparse_csr, B):
    spmm_res = torch.sparse.mm(sparse_csr, B)
    return spmm_res


def spmm_atomic(edge_index, B, C):
    module.spmm(edge_index, B, C)
    return C


def spmm_csr(rowptr, col, B, C):
    module_csr.spmm(rowptr, col, B, C)
    return C


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="spmm arg")
    # Add arguments
    parser.add_argument("-f", "--feature", type=int,
                        default=32, help="feature size")
    parser.add_argument("-n", "--nodes", type=int,
                        default=10000, help="num nodes")
    parser.add_argument("-e", "--edges", type=int,
                        default=200000, help="num edges")

    args = parser.parse_args()

    num_edges = args.edges
    num_nodes = args.nodes
    feature_size = args.feature
    nodes_feature = rand_strided((num_nodes, feature_size), (feature_size, 1),
                                 device='cpu', dtype=torch.float32)
    density = num_edges / (num_nodes * num_nodes)
    sparse = sp.random(num_nodes, num_nodes, density=density, format='coo')
    edge_index = from_scipy_sparse_matrix(sparse)[0]
    scipy_csr = sparse.tocsc()
    rowptr = torch.from_numpy(scipy_csr.indptr).to(torch.int64).to('cpu')
    col = torch.from_numpy(scipy_csr.indices).to(torch.int64).to('cpu')
    weight = torch.ones(col.shape)
    tcsr = torch.sparse_csr_tensor(
        rowptr,
        col,
        weight,
        dtype=torch.float,
        size=(num_nodes, num_nodes),
        device='cpu',
    )
    golden_res = golden_spmm(tcsr, nodes_feature)
    out_feature = torch.zeros_like(nodes_feature)
    atomic_res = spmm_atomic(edge_index, nodes_feature, out_feature)
    out_feature = torch.zeros_like(nodes_feature)
    csr_res = spmm_csr(rowptr, col, nodes_feature, out_feature)
    assert torch.allclose(golden_res, atomic_res, atol=1e-4, rtol=1e-4)
    assert torch.allclose(golden_res, csr_res, atol=1e-4, rtol=1e-4)
