from collections import defaultdict
import networkx as nx
import torch
import dgl

from local_config import PROCESSED_DATA_DIR

PARENT_EDGE_TYPE = ("node", "parent", "node")
SIBLING_EDGE_TYPE = ("node", "sibling", "node")


def get_internal_target(num_building_block_mols):
    return num_building_block_mols


def get_stop_target(num_building_block_mols):
    return num_building_block_mols + 1


def get_root_target(num_building_block_mols):
    return num_building_block_mols + 2


def save_synthetic_route_data(
    synthetic_route_graphs, synthetic_route_node2smis, synthetic_route_node2molgraphs,
):
    dgl.save_graphs(f"{PROCESSED_DATA_DIR}/synthetic_route_graphs.pt", synthetic_route_graphs)

    with open(f"{PROCESSED_DATA_DIR}/synthetic_route_node2smis.pt", "wb") as f:
        torch.save(synthetic_route_node2smis, f)

    synthetic_route_molgraphs = []
    for node2molgraph in synthetic_route_node2molgraphs:
        synthetic_route_molgraphs += node2molgraph.values()

    dgl.save_graphs(f"{PROCESSED_DATA_DIR}/synthetic_route_molgraphs.pt", synthetic_route_molgraphs)


def save_building_block_data(building_block_smis, building_block_molgraphs):
    with open(f"{PROCESSED_DATA_DIR}/building_block_smis.pt", "wb") as f:
        torch.save(building_block_smis, f)

    dgl.save_graphs(f"{PROCESSED_DATA_DIR}/building_block_molgraphs.pt", building_block_molgraphs)


def load_synthetic_route_data():
    synthetic_route_graphs, _ = dgl.load_graphs(f"{PROCESSED_DATA_DIR}/synthetic_route_graphs.pt")
    with open(f"{PROCESSED_DATA_DIR}/synthetic_route_node2smis.pt", "rb") as f:
        synthetic_route_node2smis = torch.load(f)

    synthetic_route_molgraphs, _ = dgl.load_graphs(f"{PROCESSED_DATA_DIR}/synthetic_route_molgraphs.pt")

    synthetic_route_node2molgraphs = []
    offset = 0
    for node2smi in synthetic_route_node2smis:
        nodes = list(node2smi.keys())
        molgraphs = synthetic_route_molgraphs[offset : offset + len(nodes)]
        synthetic_route_node2molgraphs.append({node: molgraph for node, molgraph in zip(nodes, molgraphs)})

    return synthetic_route_graphs, synthetic_route_node2smis, synthetic_route_node2molgraphs


def load_building_block_data():
    with open(f"{PROCESSED_DATA_DIR}/building_block_smis.pt", "rb") as f:
        building_block_smis = torch.load(f)

    building_block_molgraphs, _ = dgl.load_graphs(f"{PROCESSED_DATA_DIR}/building_block_molgraphs.pt")
    return building_block_smis, building_block_molgraphs

