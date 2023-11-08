
from torch_geometric.utils import to_scipy_sparse_matrix
import torch
from math import inf, nan

from torch import empty_strided, device
from torch.utils.cpp_extension import load
from utils import GraphDataset
from gpu.spmm_kernel import spmm_csr_wrapper, spmm_atomic_wrapper


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf0 = empty_strided(arg0_1.size(), arg0_1.stride(),
                         device='cuda', dtype=torch.float32)
    spmm_atomic_wrapper(arg1_1, arg0_1, buf0)
    del arg0_1
    del arg1_1
    return (buf0, )


def call_csr(args):
    csrptr, csrind, arg0_1 = args
    args.clear()
    buf0 = empty_strided(arg0_1.size(), arg0_1.stride(),
                         device='cuda', dtype=torch.float32)
    spmm_csr_wrapper(csrptr, csrind, arg0_1, buf0)
    del arg0_1
    del csrptr
    del csrind
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10, arg0_1=None, arg1_1=None):
    from torch._inductor.utils import print_performance

    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


def benchmark_compiled_module_csr(times=10, repeat=10, arg0_1=None, arg1_1=None, num_nodes=0):
    from torch._inductor.utils import print_performance
    scipy_coo = to_scipy_sparse_matrix(arg1_1, num_nodes=num_nodes)
    scipy_csr = scipy_coo.tocsc()
    rowptr = scipy_csr.indptr
    col = scipy_csr.indices
    weight = torch.ones(col.shape, requires_grad=True)

    tcsr = torch.sparse_csr_tensor(rowptr, col, weight, dtype=torch.float)
    csrptr = tcsr.crow_indices().to(torch.int64).to('cuda')
    csrind = tcsr.col_indices().to(torch.int64).to('cuda')
    return print_performance(lambda: call_csr([csrptr, csrind, arg0_1]), times=times, repeat=repeat)


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
                              device='cuda', dtype=torch.float32)
        arg1_1 = torch.randint(num_nodes, (2, num_edges)
                               ).to(torch.int64).to('cuda')
    else:
        dataset = GraphDataset(args.dataset, 'cuda')
        num_nodes = dataset.num_nodes
        feature_size = args.feature
        arg0_1 = rand_strided((num_nodes, feature_size), (feature_size, 1),
                              device='cuda', dtype=torch.float32)
        arg1_1 = dataset.edge_index
    benchmark_compiled_module(
        times=100, repeat=10, arg0_1=arg0_1, arg1_1=arg1_1)
    benchmark_compiled_module_csr(
        times=100, repeat=10, arg0_1=arg0_1, arg1_1=arg1_1, num_nodes=num_nodes)
