from itertools import zip_longest
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical

import networkx as nx

import dgl
import dgl.function as fn

from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph

from structure2vec import Structure2Vec

import random


def zip_discard_gen(*iterables, sentinel=object()):
    return (
        (entry for entry in iterable if entry is not sentinel)
        for iterable in zip_longest(*iterables, fillvalue=sentinel)
    )


class SyntheticTreePropagationNetwork(nn.Module):
    def __init__(
        self,
        num_vocabs,
        num_layers,
        hidden_dim,
        atom_feature_dim,
        bond_feature_dim,
        reaction_predictor,
        leaf_smis,
        leaf_graphs,
    ):
        super(SyntheticTreePropagationNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_vocabs = num_vocabs
        self.mol_embedding_layer = Structure2Vec(num_layers, hidden_dim, atom_feature_dim, bond_feature_dim)
        self.parent_layer = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.classify_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_vocabs + 2),
        )
        self.reaction_predictor = reaction_predictor
        self.leaf_smis = leaf_smis
        self.leaf_graphs = leaf_graphs
        self.atom_featurizer = CanonicalAtomFeaturizer(atom_data_field="x")
        self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field="w")

    def forward(self, graph, node2mol, device):
        nonstop_nodes = list(sorted(node2mol.keys()))
        mol_graphs = [node2mol[node][1] for node in nonstop_nodes]
        batched_mol_graph = dgl.batch(mol_graphs).to(device)
        mol_vecs = self.mol_embedding_layer(batched_mol_graph)

        graph = graph.to(device)
        graph.ndata["mol_vec"] = torch.zeros(graph.number_of_nodes(), mol_vecs.size(1)).to(device)
        graph.ndata["mol_vec"][torch.tensor(nonstop_nodes)] = mol_vecs
        graph.ndata["z"] = torch.zeros(graph.number_of_nodes(), self.hidden_dim).to(device)
        prop_traj = self.generate_prop_traj(graph)

        for cur_nodes in prop_traj:
            self.update_node_embeddings(graph, cur_nodes.to(device))

        z = graph.ndata.pop("z")
        logits = self.classify_layer(z)

        return logits

    def update_node_embeddings(self, graph, nodes):
        in_parent_edges = graph.in_edges(nodes, form="uv", etype="parent")
        graph.send_and_recv(in_parent_edges, fn.copy_src("z", "m"), fn.sum("m", "parent_z"), etype="parent")
        graph.ndata["z"][nodes] += self.parent_layer(graph.ndata["parent_z"][nodes])

        in_sibling_edges = graph.in_edges(nodes, form="uv", etype="sibling")
        if in_sibling_edges[0].size(0) > 0:
            graph.send_and_recv(
                in_sibling_edges, fn.copy_src("mol_vec", "m"), fn.sum("m", "sibling_h"), etype="sibling"
            )
            graph.ndata["z"][nodes] += graph.ndata["sibling_h"][nodes]

    def generate_prop_traj(self, graph):
        offsets = [0] + np.cumsum(graph.batch_num_nodes().cpu()[:-1]).tolist()
        prop_trajs = [
            list(range(offset, offset + num_nodes)) for offset, num_nodes in zip(offsets, graph.batch_num_nodes())
        ]
        batch_prop_traj = [torch.tensor(list(cur_nodes)) for cur_nodes in zip_discard_gen(*prop_trajs)][1:]

        return batch_prop_traj

    def get_logits(self, graph):
        return logits

    def generate(self):
        graph = dgl.graph([])
        graph.add_nodes(1)
        graph.ndata["z"] = torch.zeros(1, self.hidden_dim)
        stack = [0]
        node2smi = dict()
        node2mol_vec = dict()
        while stack and graph.number_of_nodes() < 10:
            graph, stack, node2smi, node2mol_vec = self.add_node(graph, stack, node2smi, node2mol_vec)
            if graph is None:
                return None, None

        return graph, node2smi

    def add_node(self, graph, stack, node2smi, node2mol_vec):
        parent = stack[-1]
        siblings = graph.successors(parent).tolist()
        parent_z = graph.ndata["z"][parent].unsqueeze(0)
        z = self.parent_layer(parent_z)
        for sibling in siblings:
            z += node2mol_vec[sibling] / len(siblings)

        logit = self.classify_layer(z)
        m = Categorical(logits=logit)
        decision = m.sample().squeeze(0).item()

        if decision < self.num_vocabs:
            new_node = graph.number_of_nodes()
            graph.add_nodes(1)
            graph.add_edges(parent, new_node)
            node2smi[new_node] = self.leaf_smis[decision]
            node2mol_vec[new_node] = self.mol_embedding_layer(self.leaf_graphs[decision])

        elif decision == self.num_vocabs:
            new_node = graph.number_of_nodes()
            graph.add_nodes(1)
            graph.add_edges(parent, new_node)
            graph.ndata["z"][new_node] = z
            stack.append(new_node)

        elif decision == self.num_vocabs + 1:
            stack.pop()
            child_smis = [node2smi[child] for child in siblings]
            scores, parent_smis = self.reaction_predictor.get_products([child_smis])
            node2smi[parent] = parent_smi = parent_smis[0]
            parent_graph = smiles_to_bigraph(
                parent_smi, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer
            )
            if parent_graph is None:
                return None, None, None, None

            node2mol_vec[parent] = self.mol_embedding_layer(parent_graph)

        return graph, stack, node2smi, node2mol_vec

    def _generate(self):
        self.generate_batch_size = 128
        batch_graph = dgl.heterograph({("node", "parent", "node"): [], ("node", "sibling", "node"): []})
        batch_graph.add_nodes(self.generate_batch_size)
        batch_graph.ndata["idx"] = torch.arange(self.generate_batch_size)
        batch_graph.ndata["z"] = torch.zeros(self.generate_batch_size, self.hidden_dim)
        batch_graph.ndata["mol_vec"] = torch.zeros(self.generate_batch_size, self.hidden_dim)
        batch_graph.ndata["target"] = torch.full((self.generate_batch_size, ), 0).long()
        stacks = [[idx] for idx in range(self.generate_batch_size)]
        batch_node2smi = dict()
        batch_node2graph = dict()
        failed_idxs = []
        while stacks:
            batch_graph, stacks, batch_node2smi, batch_node2graph, failed_idxs = self._add_nodes(
                batch_graph, stacks, batch_node2smi, batch_node2graph, failed_idxs
            )
        
        graphs = []
        node2smis = []
        node2graphs = []
        for idx in range(self.generate_batch_size):
            if idx not in failed_idxs:
                subgraph_is = (batch_graph.ndata["idx"] == idx).nonzero().squeeze(1).tolist()
                graphs.append(batch_graph.subgraph(subgraph_is))
                print(graphs[-1].edges(etype="parent"))
                node2smis.append({i: batch_node2smi[subgraph_i] for i, subgraph_i in enumerate(subgraph_is)})
                node2graphs.append({i: batch_node2graph[subgraph_i] for i, subgraph_i in enumerate(subgraph_is)})
        
        return graphs, node2smis, node2graphs

    def _add_nodes(self, batch_graph, stacks, batch_node2smi, batch_node2graph, failed_idxs):
        parent_list = [stack[-1] for stack in stacks if stack]
        siblings_list = [batch_graph.successors(parent, etype="parent").tolist() for parent in parent_list]

        new_node_list = list(range(batch_graph.number_of_nodes(), batch_graph.number_of_nodes() + len(stacks)))
        batch_graph.add_nodes(len(stacks))
        batch_graph.ndata["idx"][new_node_list] = batch_graph.ndata["idx"][parent_list]

        batch_graph.add_edges(parent_list, new_node_list, etype="parent")
        for siblings, new_node in zip(siblings_list, new_node_list):
            if siblings:
                batch_graph.add_edges(siblings, [new_node for _ in siblings], etype="sibling")

        self.update_node_embeddings(batch_graph, torch.tensor(new_node_list))

        new_z = batch_graph.ndata["z"][new_node_list]
        logits = self.classify_layer(new_z)
        m = Categorical(logits=logits)
        decision_tsr = m.sample()
        batch_graph.ndata["target"][new_node_list] = decision_tsr

        decision_list = decision_tsr.tolist()
        reactants_list, product_node_list, product_stacks = [], [], []
        new_graphs, nodes_with_new_graphs = [], []

        for stack, new_node, siblings, decision in zip(stacks, new_node_list, siblings_list, decision_list):
            if decision < self.num_vocabs:
                batch_node2smi[new_node] = self.leaf_smis[decision]
                batch_node2graph[new_node] = self.leaf_graphs[decision]
                
                nodes_with_new_graphs.append(new_node)
                new_graphs.append(self.leaf_graphs[decision])

            elif decision == self.num_vocabs:
                stack.append(new_node)

            elif decision == self.num_vocabs + 1:
                if siblings:
                    parent = stack.pop()
                    product_node_list.append(parent)
                    product_stacks.append(stack)
                    reactants_list.append([batch_node2smi[child] for child in siblings])
                    batch_node2smi[new_node] = None
                    batch_node2graph[new_node] = None
                else:
                    stack.clear()
                    failed_idxs.append(batch_graph.ndata["idx"][new_node].item())

        if reactants_list:
            scores, product_smi_list = self.reaction_predictor.get_products(reactants_list)
            for product_node, stack, product_smi in zip(product_node_list, product_stacks, product_smi_list):
                if product_smi is None:
                    failed_idxs.append(batch_graph.ndata["idx"][product_node].item())
                    stack.clear()
                else:
                    product_graph = smiles_to_bigraph(
                        product_smi, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer
                        )

                    batch_node2smi[product_node] = product_smi
                    batch_node2graph[product_node] = product_graph

                    nodes_with_new_graphs.append(product_node)
                    new_graphs.append(product_graph)

        if new_graphs:
            batch_mol_graph = dgl.batch(new_graphs)
            batch_mol_vec = self.mol_embedding_layer(batch_mol_graph)
            batch_graph.ndata["mol_vec"][nodes_with_new_graphs] = batch_mol_vec

        stacks = [stack for stack in stacks if stack]
        
        return batch_graph, stacks, batch_node2smi, batch_node2graph, failed_idxs
