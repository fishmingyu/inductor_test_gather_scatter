
import argparse
from torch_geometric.utils import from_scipy_sparse_matrix
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
import scipy.sparse as sp
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from torch._inductor.select_algorithm import extern_kernels
from torch.utils.cpp_extension import load
from torch._dynamo.testing import rand_strided

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the parent directory of current_dir
parent_dir = os.path.dirname(current_dir)
# Get the path to subpackage1
subpackage1_dir = os.path.join(parent_dir, 'gpu')

# Add subpackage1 to sys.path
sys.path.insert(0, subpackage1_dir)
from spmm_kernel import spmm_atomic_wrapper, spmm_csr_wrapper

def golden_spmm(sparse_csr, B):
    spmm_res = torch.sparse.mm(sparse_csr, B)
    return spmm_res


def spmm_atomic(edge_index, B, C):
    spmm_atomic_wrapper(edge_index, B, C)
    return C


def spmm_csr(rowptr, col, B, C):
    spmm_csr_wrapper(rowptr, col, B, C)
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
                                 device='cuda', dtype=torch.float32)
    density = num_edges / (num_nodes * num_nodes)
    sparse = sp.random(num_nodes, num_nodes, density=density, format='coo')
    edge_index = from_scipy_sparse_matrix(sparse)[0].to("cuda")
    scipy_csr = sparse.tocsc()
    rowptr = torch.from_numpy(scipy_csr.indptr).to(torch.int64).to('cuda')
    col = torch.from_numpy(scipy_csr.indices).to(torch.int64).to('cuda')
    weight = torch.ones(col.shape)
    tcsr = torch.sparse_csr_tensor(
        rowptr,
        col,
        weight,
        dtype=torch.float,
        size=(num_nodes, num_nodes),
        device='cuda',
    )

    golden_res = golden_spmm(tcsr, nodes_feature)
    atomic_res = torch.zeros_like(nodes_feature)
    spmm_atomic(edge_index, nodes_feature, atomic_res)
    out_feature = torch.zeros_like(nodes_feature)
    csr_res = spmm_csr(rowptr, col, nodes_feature, out_feature)
    assert torch.allclose(golden_res, atomic_res, atol=1e-4, rtol=1e-4)
    assert torch.allclose(golden_res, csr_res, atol=1e-4, rtol=1e-4)
