from itertools import zip_longest
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import dgl

from data.synthetic_route import load_synthetic_route_data


class SyntheticRouteDataset(Dataset):
    def __init__(self):
        super(SyntheticRouteDataset, self).__init__()
        (
            self.synthetic_route_graphs,
            self.synthetic_route_node2smis,
            self.synthetic_route_node2molgraphs,
        ) = load_synthetic_route_data()

    def __len__(self):
        return len(self.synthetic_route_graphs)

    def __getitem__(self, idx):
        graph = self.synthetic_route_graphs[idx]
        node2smi = self.synthetic_route_node2smis[idx]
        node2molgraph = self.synthetic_route_node2molgraphs[idx]
        traversal_order = self._gen_traversal_order(graph)

        return graph, node2smi, node2molgraph, traversal_order

    def _gen_traversal_order(self, graph):
        return torch.arange(graph.number_of_nodes())


def zip_discard_gen(*iterables, sentinel=object()):
    return (
        (entry for entry in iterable if entry is not sentinel)
        for iterable in zip_longest(*iterables, fillvalue=sentinel)
    )


def synthetic_route_collate_fn(batch):
    graphs, node2smis, node2molgraphs, traversal_orders = zip(*batch)

    batch_graph = dgl.batch(graphs)
    batch_graph.set_n_initializer(dgl.init.zero_initializer)

    offset = 0
    batch_node2molgraph = dict()
    batch_node2smi = dict()
    traversal_orders_with_offsets = []
    for node2molgraph, node2smi, traversal_order, num_nodes in zip(
        node2molgraphs, node2smis, traversal_orders, batch_graph.batch_num_nodes().tolist()
    ):
        batch_node2molgraph.update({node + offset: mol for node, mol in node2molgraph.items()})
        batch_node2smi.update({node + offset: smi for node, smi in node2smi.items()})
        traversal_orders_with_offsets.append((traversal_order + offset).tolist())
        offset += num_nodes

    batch_traversal_orders = list(
        [entry for entry in iterable if entry is not None]
        for iterable in zip_longest(*traversal_orders_with_offsets, fillvalue=None)
    )[1:]
    return batch_graph, batch_node2smi, batch_node2molgraph, batch_traversal_orders


if __name__ == "__main__":
    from tqdm import tqdm

    a = []
    dataset = SyntheticRouteDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, collate_fn=synthetic_route_collate_fn)
    for graph, node2molgraph, node2smi, traversal_orders in tqdm(loader):
        pass
        # print(list(traversal_orders))
