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
        scipy_coo = to_scipy_sparse_matrix(graph.edge_index,
                                           num_nodes=graph.num_nodes)
        # print(dgl_graph)
        # print(dgl_graph.adj())
        # print(dgl_graph.adj().val.sum())
        scipy_csr = scipy_coo.tocsc()
        rowptr = torch.from_numpy(scipy_csr.indptr).to(torch.int32).to(
            self.device)
        self.rowptr = rowptr
        col = torch.from_numpy(scipy_csr.indices).to(torch.int32).to(
            self.device)
        self.col = col
        weight = torch.ones(col.shape, requires_grad=True)
        self.num_nodes = graph.num_nodes
        self.tcsr = torch.sparse_csr_tensor(
            rowptr,
            col,
            weight,
            dtype=torch.float,
            size=(self.num_nodes, self.num_nodes),
            requires_grad=True,
            device=self.device,
        )
        self.num_edges = graph.num_edges

        self.features = graph.x.to(self.device)
