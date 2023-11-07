import os
import os.path as osp
import warnings

import pytest
import torch
import torch.nn.functional as F

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.models import GAT, GCN, GIN, PNA, EdgeCNN, GraphSAGE
from torch_geometric.profile import benchmark
from torch_geometric.testing import (
    disableExtensions,
    onlyFullTest,
    onlyLinux,
    withCUDA,
    withPackage,
)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    x = torch.randn(num_nodes, 64, device=args.device)
    edge_index = torch.randint(num_nodes, (2, num_edges), device=args.device)

    for Model in [GCN, GraphSAGE, GIN, GAT]:
        print(f'Model: {Model.__name__}')

        model = Model(64, 64, num_layers=3).to(args.device)
        compiled_model = torch_geometric.compile(model)

        benchmark(
            funcs=[model, compiled_model],
            func_names=['Vanilla', 'Compiled'],
            args=(x, edge_index),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )
