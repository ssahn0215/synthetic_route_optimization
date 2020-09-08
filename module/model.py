import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical

import networkx as nx

import dgl
import dgl.function as fn

from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph

from module.structure2vec import Structure2VecFirstLayer, Structure2VecLayer
from module.reaction_predictor import ReactionPredicter
from data.synthetic_route import PARENT_EDGE_TYPE, SIBLING_EDGE_TYPE


class SyntheticRouteEmbeddingNetwork(nn.Module):
    def __init__(self, num_embedding_layers, hidden_dim, input_node_dim, input_edge_dim, dropout=0):
        super(SyntheticRouteEmbeddingNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.first_layer = Structure2VecFirstLayer(hidden_dim, input_node_dim, input_edge_dim, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                Structure2VecLayer(hidden_dim, input_node_dim, input_edge_dim, dropout=dropout)
                for _ in range(num_embedding_layers)
            ]
        )
        self.last_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim))

    def readout(self, g, features):
        g.ndata["h"] = features
        h = dgl.readout_nodes(g, "h", op="mean")
        return h

    def forward(self, g, device=None):
        features = self.first_layer(g)
        for layer in self.layers:
            features = layer(g, features)
        features = self.last_layer(features)
        features = self.readout(g, features)

        return features


class SyntheticRoutePropagationNetwork(nn.Module):
    def __init__(self, hidden_dim):
        super(SyntheticRoutePropagationNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, graph, prop_nodes):
        in_parent_edges = graph.in_edges(prop_nodes, form="uv", etype=PARENT_EDGE_TYPE)
        graph.send_and_recv(
            in_parent_edges, fn.copy_src("prop_z", "m"), fn.sum("m", "parent_prop_z"), etype=PARENT_EDGE_TYPE
        )
        parent_prop_z = graph.ndata.pop("parent_prop_z")
        graph.ndata["prop_z"][prop_nodes] += self.relu(self.linear(parent_prop_z[prop_nodes]))

        in_sibling_edges = graph.in_edges(prop_nodes, form="uv", etype=SIBLING_EDGE_TYPE)
        if in_sibling_edges[0].size(0) > 0:
            graph.send_and_recv(
                in_sibling_edges, fn.copy_src("mol_z", "m"), fn.sum("m", "sibling_prop_z"), etype=SIBLING_EDGE_TYPE
            )
            sibling_prop_z = graph.ndata.pop("sibling_prop_z")
            graph.ndata["prop_z"][prop_nodes] += sibling_prop_z[prop_nodes]

        return graph


class SyntheticRouteClassificationNetwork(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(SyntheticRouteClassificationNetwork, self).__init__()
        self.bn0 = nn.BatchNorm1d(hidden_dim)
        self.relu0 = nn.ReLU()
        self.linear0 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        out = self.linear0(self.relu0(self.bn0(h)))
        out = self.linear1(self.relu1(self.bn1(h)))
        return out


class SyntheticRouteNetwork(nn.Module):
    def __init__(self, num_embedding_layers, hidden_dim, input_node_dim, input_edge_dim, output_dim):
        super(SyntheticRouteNetwork, self).__init__()
        self.input_node_dim = input_node_dim
        self.input_edge_dim = input_edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding_layer = SyntheticRouteEmbeddingNetwork(
            num_embedding_layers, hidden_dim, input_node_dim, input_edge_dim
        )
        self.propagation_layer = SyntheticRoutePropagationNetwork(hidden_dim)
        self.classification_layer = SyntheticRouteClassificationNetwork(hidden_dim, output_dim)
        self.reaction_predictor = ReactionPredicter()

    def get_default_mol_z(self, num_nodes):
        return torch.zeros(num_nodes, self.hidden_dim)

    def get_default_prop_z(self, num_nodes):
        return torch.zeros(num_nodes, self.hidden_dim)
