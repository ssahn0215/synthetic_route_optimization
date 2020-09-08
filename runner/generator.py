import torch
from torch.distributions import Categorical
import dgl
from data.molecule import smi2molgraph
from data.synthetic_route import (
    PARENT_EDGE_TYPE,
    SIBLING_EDGE_TYPE,
    get_internal_target,
    get_stop_target,
    get_root_target,
)


class SyntheticRouteGenerator:
    def __init__(
        self, model, building_block_smis, building_block_molgraphs, batch_size, device,
    ):
        self.model = model
        self.building_block_smis = building_block_smis
        self.building_block_molgraphs = building_block_molgraphs
        self.batch_size = batch_size
        self.device = device

        num_building_block_smis = len(self.building_block_smis)
        self.INTERNAL_TARGET = get_internal_target(num_building_block_smis)
        self.STOP_TARGET = get_stop_target(num_building_block_smis)
        self.ROOT_TARGET = get_root_target(num_building_block_smis)

    def generate(self, num_samples):
        graphs, node2smis, node2molgraphs = [], [], []
        offset = 0
        while offset < num_samples:
            num_samples = min(self.batch_size, num_samples - offset)
            offset += num_samples

            batch_graphs, batch_node2smis, batch_node2molgraphs = self.generate_batch(num_samples)
            graphs += batch_graphs
            node2smis += batch_node2smis
            node2molgraphs += batch_node2molgraphs

        return graphs, node2smis, node2molgraphs

    def generate_batch(self, num_samples):
        batch_graph = dgl.heterograph({PARENT_EDGE_TYPE: [], SIBLING_EDGE_TYPE: []})
        batch_graph.add_nodes(num_samples)
        batch_graph.ndata["subgraph_idx"] = torch.arange(num_samples)
        batch_graph.ndata["prop_z"] = self.model.get_default_prop_z(num_samples)
        batch_graph.ndata["mol_z"] = self.model.get_default_mol_z(num_samples)
        batch_graph.ndata["target"] = torch.full((num_samples,), self.ROOT_TARGET).long()
        batch_graph = batch_graph.to(self.device)
        stacks = [[idx] for idx in range(num_samples)]
        batch_node2smi = dict()
        batch_node2molgraph = dict()
        failed_subgraph_idxs = []

        for _ in range(10):
            batch_graph, stacks, batch_node2smi, batch_node2molgraph, failed_subgraph_idxs = self.add_nodes(
                batch_graph, stacks, batch_node2smi, batch_node2molgraph, failed_subgraph_idxs
            )

            active_stacks = [stack for idx, stack in enumerate(stacks) if (idx not in failed_subgraph_idxs) and stack]
            if not active_stacks:
                break

        batch_graph.ndata.pop("prop_z")
        batch_graph.ndata.pop("mol_z")
        subgraph_idxs = batch_graph.ndata.pop("subgraph_idx")

        graphs = []
        node2smis = []
        node2molgraphs = []
        for idx in range(num_samples):
            if idx in failed_subgraph_idxs:
                continue

            inducing_nodes = (subgraph_idxs == idx).nonzero().squeeze(1).tolist()
            subgraph = batch_graph.subgraph(inducing_nodes)

            subgraph_node2smi = {
                subgraph_node: batch_node2smi[inducing_node]
                for subgraph_node, inducing_node in enumerate(inducing_nodes)
                if inducing_node in batch_node2smi
            }

            subgraph_node2molgraphs = {
                subgraph_node: batch_node2molgraph[inducing_node]
                for subgraph_node, inducing_node in enumerate(inducing_nodes)
                if inducing_node in batch_node2smi
            }

            graphs.append(subgraph)
            node2smis.append(subgraph_node2smi)
            node2molgraphs.append(subgraph_node2molgraphs)

        return graphs, node2smis, node2molgraphs

    def add_nodes(self, batch_graph, stacks, batch_node2smi, batch_node2molgraph, failed_subgraph_idxs):
        active_stacks = [stack for idx, stack in enumerate(stacks) if (idx not in failed_subgraph_idxs) and stack]

        parent_list = [stack[-1] for stack in active_stacks]
        added_nodes = list(range(batch_graph.number_of_nodes(), batch_graph.number_of_nodes() + len(active_stacks)))

        batch_graph.add_nodes(len(active_stacks))
        batch_graph.ndata["subgraph_idx"][added_nodes] = batch_graph.ndata["subgraph_idx"][parent_list]
        siblings_list = [batch_graph.successors(parent, etype=PARENT_EDGE_TYPE).tolist() for parent in parent_list]
        for siblings, added_node in zip(siblings_list, added_nodes):
            if siblings:
                batch_graph.add_edges(siblings, [added_node for _ in siblings], etype=SIBLING_EDGE_TYPE)

        batch_graph.add_edges(parent_list, added_nodes, etype=PARENT_EDGE_TYPE)
        batch_graph = self.model.propagation_layer(batch_graph, added_nodes)

        added_prop_z = batch_graph.ndata["prop_z"][added_nodes]
        logits = self.model.classification_layer(added_prop_z)
        m = Categorical(logits=logits)
        decisions_tensor = m.sample()
        batch_graph.ndata["target"][added_nodes] = decisions_tensor

        decisions = decisions_tensor.tolist()
        completed_node2molgraph, completed_node2smi = dict(), dict()
        reactants_list, reaction_nodes = [], []
        for stack, added_node, siblings, decision in zip(active_stacks, added_nodes, siblings_list, decisions):
            if decision < self.INTERNAL_TARGET:
                completed_node2smi[added_node] = self.building_block_smis[decision]
                completed_node2molgraph[added_node] = self.building_block_molgraphs[decision]

            elif decision == self.INTERNAL_TARGET:
                stack.append(added_node)

            elif decision == self.STOP_TARGET:
                if not siblings:
                    subgraph_idx = batch_graph.ndata["subgraph_idx"][added_node].item()
                    failed_subgraph_idxs.append(subgraph_idx)
                    continue

                parent = stack.pop()
                reaction_nodes.append(parent)
                reactants_list.append([batch_node2smi[child] for child in siblings])

        if reactants_list:
            scores, product_smi_list = self.model.reaction_predictor.get_products(reactants_list)
            for reaction_node, product_smi in zip(reaction_nodes, product_smi_list):
                if product_smi is None:
                    subgraph_idx = batch_graph.ndata["subgraph_idx"][reaction_node].item()
                    failed_subgraph_idxs.append(subgraph_idx)
                    continue

                product_graph = smi2molgraph(product_smi)
                completed_node2smi[reaction_node] = product_smi
                completed_node2molgraph[reaction_node] = product_graph

        if completed_node2smi:
            completed_nodes = list(completed_node2smi.keys())
            batch_completed_molgraph = dgl.batch([completed_node2molgraph[node] for node in completed_nodes])
            batch_completed_molgraph = batch_completed_molgraph.to(self.device)
            batch_completed_mol_embedding = self.model.embedding_layer(batch_completed_molgraph)
            batch_graph.ndata["mol_z"][completed_nodes] = batch_completed_mol_embedding

        batch_node2smi.update(completed_node2smi)
        batch_node2molgraph.update(completed_node2molgraph)

        return batch_graph, stacks, batch_node2smi, batch_node2molgraph, failed_subgraph_idxs

