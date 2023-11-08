import torch_geometric.datasets as datasets
from torch_geometric.utils import to_scipy_sparse_matrix
import torch


class GraphDataset:

    def __init__(self, name: str, device):
        self.name = name
        self.device = device
        self.get_dataset()

    def get_dataset(self):
        if self.name == 'pubmed':
            dataset = datasets.Planetoid(root='./data/', name='Pubmed')
            graph = dataset[0]
        elif self.name == 'citeseer':
            dataset = datasets.Planetoid(root='./data/', name='Citeseer')
            graph = dataset[0]
        elif self.name == 'cora':
            dataset = datasets.Planetoid(root='./data/', name='Cora')
            graph = dataset[0]
        elif self.name == 'ppi':
            dataset = datasets.PPI(root='./data/')
            graph = dataset[0]
        elif self.name == 'reddit':
            dataset = datasets.Reddit(root='./data/Reddit')
            graph = dataset[0]
        elif self.name == 'github':
            dataset = datasets.GitHub(root='./data/')
            graph = dataset[0]
        else:
            raise KeyError('Unknown dataset {}.'.format(self.name))
        self.edge_index = graph.edge_index.to(self.device)
        self.num_edges = graph.num_edges
        self.num_nodes = graph.num_nodes

        self.features = graph.x.to(self.device)
