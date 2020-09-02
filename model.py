from itertools import zip_longest
import numpy as np

import torch
import torch.nn as nn

import dgl
import dgl.function as fn

from structure2vec import Structure2Vec

def zip_discard_gen(*iterables, sentinel=object()):
    return (
        (entry for entry in iterable if entry is not sentinel)
        for iterable in zip_longest(*iterables, fillvalue=sentinel)
    )


class SyntheticTreePropagationNetwork(nn.Module):
    def __init__(self, num_vocabs, num_layers, hidden_dim, atom_feature_dim, bond_feature_dim):
        super(SyntheticTreePropagationNetwork, self).__init__()
        self.mol_embedding_layer = Structure2Vec(num_layers, hidden_dim, atom_feature_dim, bond_feature_dim)
        self.parent_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.classify_layer = nn.Linear(hidden_dim, num_vocabs+2)

    def forward(self, graph, node2mol, device):
        nonstop_nodes = list(sorted(node2mol.keys()))
        mol_graphs = [node2mol[node][1] for node in nonstop_nodes]
        batched_mol_graph = dgl.batch(mol_graphs).to(device)
        mol_vecs = self.mol_embedding_layer(batched_mol_graph)
        
        graph = graph.to(device)
        graph.ndata["mol_vec"] = torch.zeros(graph.number_of_nodes(), mol_vecs.size(1)).to(device)
        graph.ndata["mol_vec"][torch.tensor(nonstop_nodes)] = mol_vecs
        graph.ndata["z"] = torch.zeros_like(graph.ndata["mol_vec"])
        prop_traj = self.generate_prop_traj(graph)
        for cur_nodes in prop_traj[1:]:
            self.update_node_embeddings(graph, cur_nodes.to(device))

        logits = self.get_logits(graph)

        return logits

    def update_node_embeddings(self, graph, nodes):
        in_parent_edges = graph.in_edges(nodes, form="eid", etype="parent")
        graph.send_and_recv(in_parent_edges, fn.copy_src("z", "m"), fn.sum("m", "parent_z"), etype="parent")
        graph.ndata["z"][nodes] += self.parent_layer(graph.ndata["parent_z"][nodes])

        in_sibling_edges = graph.in_edges(nodes, form="eid", etype="sibling")
        if in_sibling_edges.size(0) > 0:
            graph.send_and_recv(
                in_sibling_edges, fn.copy_src("mol_vec", "m"), fn.mean("m", "sibling_h"), etype="sibling"
            )
            graph.ndata["z"][nodes] += graph.ndata["sibling_h"][nodes]

    def generate_prop_traj(self, graph):
        offsets = [0] + np.cumsum(graph.batch_num_nodes().cpu()[:-1]).tolist()
        prop_trajs = [
            list(range(offset, offset + num_nodes)) for offset, num_nodes in zip(offsets, graph.batch_num_nodes())
        ]
        batch_prop_traj = [torch.tensor(list(cur_nodes)) for cur_nodes in zip_discard_gen(*prop_trajs)]
        return batch_prop_traj

    def get_logits(self, graph):
        z = graph.ndata.pop("z")
        logits = self.classify_layer(z)
        return logits


