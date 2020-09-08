import pickle
from collections import defaultdict, OrderedDict
from multiprocessing import Pool
import networkx as nx
import torch
import dgl

import os, sys, inspect
from tqdm import tqdm
from copy import deepcopy

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from local_config import PROCESSED_DATA_DIR
from data.molecule import smi2molgraph
from data.synthetic_route import (
    PARENT_EDGE_TYPE,
    SIBLING_EDGE_TYPE,
    get_internal_target,
    get_root_target,
    get_stop_target,
    save_synthetic_route_data,
    save_building_block_data,
)


def build_synthetic_routes(reactions):
    product2reactants_list = defaultdict(list)
    smi2ancestor_smis = defaultdict(list)
    is_parent_smi = defaultdict(bool)
    is_child_smi = defaultdict(bool)
    seen_smis = set()
    removed_reaction_idxs = []
    for idx, (reactants, product) in enumerate(reactions):
        if set(smi2ancestor_smis[product]).intersection(set(reactants)):
            continue

        seen_smis.add(product)
        product2reactants_list[product].append(reactants)
        is_parent_smi[product] = True
        for reactant in reactants:
            seen_smis.add(reactant)
            smi2ancestor_smis[reactant] += [product] + smi2ancestor_smis[product]
            is_child_smi[reactant] = True

    seen_smis = list(sorted(seen_smis))
    roots = [smi for smi in seen_smis if not is_child_smi[smi]]
    building_block_smis = [smi for smi in seen_smis if not is_parent_smi[smi]]

    num_building_block_smis = len(building_block_smis)
    INTERNAL_TARGET = get_internal_target(num_building_block_smis)
    STOP_TARGET = get_stop_target(num_building_block_smis)
    ROOT_TARGET = get_root_target(num_building_block_smis)
    
    def _expand_parent(graph_sizes, smi2nodes, edge_dicts, parent_smi):
        expanded_graph_sizes, expanded_smi2nodes, expanded_edge_dicts = [], [], []
        for reactants in product2reactants_list[parent_smi]:
            cur_graph_sizes, cur_smi2nodes, cur_edge_dicts = deepcopy([graph_sizes, smi2nodes, edge_dicts])
            sibling_smis = []
            for reactant in list(reactants) + [None]:
                for idx in range(len(cur_graph_sizes)):
                    new_child = cur_graph_sizes[idx]

                    cur_graph_sizes[idx] += 1

                    cur_edge_dicts[idx][PARENT_EDGE_TYPE].append([cur_smi2nodes[idx][parent_smi], new_child])
                    for sibling_smi in sibling_smis:
                        cur_edge_dicts[idx][SIBLING_EDGE_TYPE].append([cur_smi2nodes[idx][sibling_smi], new_child])
                        #cur_edge_dicts[idx][SIBLING_EDGE_TYPE].append([new_child, cur_smi2nodes[idx][sibling_smi]])

                    if reactant is not None:
                        cur_smi2nodes[idx][reactant] = new_child

                if reactant is not None:
                    sibling_smis.append(reactant)
                    if reactant in product2reactants_list:
                        cur_graph_sizes, cur_smi2nodes, cur_edge_dicts = _expand_parent(
                            cur_graph_sizes, cur_smi2nodes, cur_edge_dicts, reactant
                        )

            expanded_graph_sizes += cur_graph_sizes
            expanded_smi2nodes += cur_smi2nodes
            expanded_edge_dicts += cur_edge_dicts

        return expanded_graph_sizes, expanded_smi2nodes, expanded_edge_dicts

    synthetic_route_smi2nodes = []
    synthetic_route_edge_dicts = []
    for root in tqdm(roots):
        _, smi2nodes, edge_dicts = _expand_parent(
            [1], [{root: 0}], [{PARENT_EDGE_TYPE: [], SIBLING_EDGE_TYPE: []}], root
        )
        synthetic_route_smi2nodes += smi2nodes
        synthetic_route_edge_dicts += edge_dicts

    synthetic_route_graphs = []
    synthetic_route_node2smis = []
    for edge_dict, smi2node in tqdm(list(zip(synthetic_route_edge_dicts, synthetic_route_smi2nodes))):
        graph = dgl.heterograph(edge_dict)
        node2smi = {node: smi for smi, node in smi2node.items()}
        targets = [ROOT_TARGET]
        for node in range(1, graph.number_of_nodes()):
            smi = node2smi.get(node, None)
            if smi is not None:
                if smi in building_block_smis:
                    targets.append(building_block_smis.index(smi))
                else:
                    targets.append(INTERNAL_TARGET)
            else:
                targets.append(STOP_TARGET)

        graph.ndata["target"] = torch.tensor(targets)

        synthetic_route_graphs.append(graph)
        synthetic_route_node2smis.append(node2smi)

    return synthetic_route_graphs, synthetic_route_node2smis, building_block_smis


if __name__ == "__main__":
    reactions = []
    for split in ["train", "val", "test"]:
        processed_reactions_filename = f"{PROCESSED_DATA_DIR}/processed_{split}.pkl"
        with open(processed_reactions_filename, "rb") as f:
            reactions += pickle.load(f)

    #reactions = reactions[:100]


    synthetic_route_graphs, synthetic_route_node2smis, building_block_smis = build_synthetic_routes(reactions)

    synthetic_route_node2molgraphs = [
        {node: smi2molgraph(smi) for node, smi in node2smi.items()} for node2smi in synthetic_route_node2smis
    ]
    building_block_molgraphs = [smi2molgraph(smi) for smi in building_block_smis]

    save_synthetic_route_data(
        synthetic_route_graphs, synthetic_route_node2smis, synthetic_route_node2molgraphs,
    )
    save_building_block_data(building_block_smis, building_block_molgraphs)

