import torch
from torch.utils.data import Dataset, DataLoader

import dgl

from copy import deepcopy
from collections import defaultdict
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_predecessors

from local_config import PROCESSED_DATA_DIR


def load_molecules():
    with open(f"{PROCESSED_DATA_DIR}/vocab_smis.pt", "rb") as f:
        vocab_smis = torch.load(f)
    with open(f"{PROCESSED_DATA_DIR}/vocab_graphs.pt", "rb") as f:
        vocab_graphs = torch.load(f)
    with open(f"{PROCESSED_DATA_DIR}/leaf_smis.pt", "rb") as f:
        leaf_smis = torch.load(f)
    with open(f"{PROCESSED_DATA_DIR}/leaf_graphs.pt", "rb") as f:
        leaf_graphs = torch.load(f)

    return vocab_smis, vocab_graphs, leaf_smis, leaf_graphs


def load_trees():
    with open(f"{PROCESSED_DATA_DIR}/synth_trees.pt", "rb") as f:
        trees = torch.load(f)

    return trees


MP_NODE_LEAF = 0
MP_NODE_STOP = 1
MP_NODE_NONLEAF = 2
MP_EDGE_PARENT = 0
MP_EDGE_SIBLING = 1


class SyntheticTreeDataset(Dataset):
    def __init__(self):
        super(SyntheticTreeDataset, self).__init__()
        self.trees = load_trees()
        self.vocab_smis, self.vocab_graphs, self.leaf_smis, self.leaf_graphs = load_molecules()

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, idx):
        tree = self.trees[idx]
        mp_graph, mp_node2mol = self.construct_mp_graph(tree)
        return mp_graph, mp_node2mol

    def construct_mp_graph(self, g):
        assert nx.algorithms.tree.recognition.is_arborescence(g)

        num_nodes = g.number_of_nodes()
        node2smi = nx.get_node_attributes(g, "smi")
        nonleaf_target = len(self.leaf_smis)
        stop_target = len(self.leaf_smis) + 1

        mp_edge_dict = defaultdict(list)
        targets = [nonleaf_target]
        mp_node_types = [MP_NODE_NONLEAF]
        mp_edge_types = []

        # We assume root has 0 idx
        stack = [(0, 0, iter(g[0]))]
        seen_childs = defaultdict(list)
        mp_child = 0
        root_smi = node2smi[0]
        mp_node2mol = {0: (root_smi, self.vocab_graphs[self.vocab_smis.index(root_smi)])}
        while stack:
            parent, mp_parent, childs_iter = stack[-1]
            mp_child += 1
            child = next(childs_iter, None)
            if child is not None:
                smi = node2smi[child]
                mp_node2mol[mp_child] = (smi, self.vocab_graphs[self.vocab_smis.index(smi)])

            if child is None:
                mp_node_types.append(MP_NODE_STOP)
                stack.pop()
                targets.append(stop_target)
            elif g.out_degree[child] > 0:
                mp_node_types.append(MP_NODE_NONLEAF)
                stack.append((child, mp_child, iter(g[child])))
                targets.append(nonleaf_target)
            else:
                mp_node_types.append(MP_NODE_LEAF)
                targets.append(self.leaf_smis.index(node2smi[child]))

            mp_edge_dict[("node", "parent", "node")].append(torch.tensor([mp_parent, mp_child]))
            mp_edge_types.append(MP_EDGE_PARENT)
            for sibling, mp_sibling in seen_childs[parent]:
                mp_edge_dict[("node", "sibling", "node")].append(torch.tensor([mp_sibling, mp_child]))
                mp_edge_types.append(MP_EDGE_SIBLING)

            seen_childs[parent].append((child, mp_child))

        mp_graph = dgl.heterograph(mp_edge_dict)
        mp_graph.ndata["y"] = torch.tensor(targets)
        mp_graph.ndata["type"] = torch.tensor(mp_node_types)

        return mp_graph, mp_node2mol



def synthetic_tree_collate_fn(batch):
    mp_graphs, mp_node2mols = zip(*batch)

    batch_mp_graph = dgl.batch(mp_graphs)
    batch_mp_graph.set_n_initializer(dgl.init.zero_initializer)

    offset = 0
    batch_mp_node2mol = dict()
    for mp_node2mol, num_nodes in zip(mp_node2mols, batch_mp_graph.batch_num_nodes()):
        batch_mp_node2mol.update({mp_node + offset: mol for mp_node, mol in mp_node2mol.items()})
        offset += num_nodes

    return batch_mp_graph, batch_mp_node2mol


if __name__ == "__main__":
    from tqdm import tqdm

    a = []
    dataset = SyntheticTreeDataset()
    mp_graph, mp_node2mol = dataset[0]
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=synthetic_tree_collate_fn)
    for batch in tqdm(loader):
        batch = batch
