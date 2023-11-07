import torch

import torch_geometric
from torch_geometric.profile import benchmark
from torch_geometric.testing import (
    disableExtensions,
    onlyFullTest,
    onlyLinux,
    withCUDA,
    withPackage,
)
from torch_geometric.utils import scatter


# Basic "Gather-Apply-Scatter" patterns commonly used in PyG:
def gather_scatter(x, edge_index, reduce="sum"):
    row, col = edge_index
    x_j = x[row]
    return scatter(x_j, col, dim_size=x.size(0), reduce=reduce)


@onlyLinux
@onlyFullTest
@disableExtensions
@withPackage("torch>=2.0.0")
def test_torch_compile(device):
    x = torch.randn(10, 16, device=device)
    edge_index = torch.randint(0, x.size(0), (2, 40), device=device)
    edge_weight = torch.rand(edge_index.size(1), device=device)
    matrix = torch.randn(x.size(-1), x.size(-1), device=device)

    expected = gather_scatter(x, edge_index)
    compiled_op = torch_geometric.compile(gather_scatter)
    out = compiled_op(x, edge_index)
    assert torch.allclose(out, expected, atol=1e-6)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    feature_size = 32
    x = torch.randn(num_nodes, feature_size, device=args.device)
    edge_index = torch.randint(num_nodes, (2, num_edges), device=args.device)
    edge_weight = torch.rand(num_edges, device=args.device)
    matrix = torch.randn(feature_size, feature_size, device=args.device)

    compiled_func = torch_geometric.compile(gather_scatter, backend="inductor")
    compiled_func(x, edge_index)
