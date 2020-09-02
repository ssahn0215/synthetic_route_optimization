import pickle
from collections import defaultdict
from multiprocessing import Pool
import networkx as nx
import torch
import dgl
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tqdm import tqdm

from local_config import PROCESSED_DATA_DIR

if __name__ == "__main__":
    reactions = []
    for split in ["train", "val", "test"]:
        processed_reactions_filename = f"{PROCESSED_DATA_DIR}/processed_{split}.pkl"
        with open(processed_reactions_filename, "rb") as f:
            reactions += pickle.load(f)

    reactions = reactions[:1000]

    vocab_smis = set()
    parent2childs_list = defaultdict(list)
    is_parent = defaultdict(bool)
    is_child = defaultdict(bool)
    for reactants, product in reactions:
        parent2childs_list[product].append(reactants)
        is_parent[product] = True
        for reactant in reactants:
            is_child[reactant] = True

        for smi in reactants + (product,):
            vocab_smis.add(smi)

    vocab_smis = sorted(list(vocab_smis))
    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field="x")
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field="w")
    vocab_graphs = []
    for smi in tqdm(vocab_smis):
        vocab_graphs.append(smiles_to_bigraph(smi, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer))

    leaf_idxs = list(filter(lambda idx: not is_parent[vocab_smis[idx]], range(len(vocab_smis))))
    leaf_smis = [vocab_smis[idx] for idx in leaf_idxs]
    leaf_graphs = [vocab_graphs[idx] for idx in leaf_idxs]

    with open(f"{PROCESSED_DATA_DIR}/vocab_smis.pt", "wb") as f:
        torch.save(vocab_smis, f)
    with open(f"{PROCESSED_DATA_DIR}/vocab_graphs.pt", "wb") as f:
        torch.save(vocab_graphs, f)
    with open(f"{PROCESSED_DATA_DIR}/leaf_smis.pt", "wb") as f:
        torch.save(leaf_smis, f)
    with open(f"{PROCESSED_DATA_DIR}/leaf_graphs.pt", "wb") as f:
        torch.save(leaf_graphs, f)

    def _recursively_expand_tree(parent_id, parent, tree):
        child_depth = tree.nodes[parent_id]["depth"] + 1
        seen = list(nx.get_node_attributes(tree, "smi").values())

        expanded_trees = []
        for childs in parent2childs_list[parent]:
            cur_tree = tree.copy()
            for child in childs:
                if vocab_smis.index(child) in seen:
                    break

                child_id = cur_tree.number_of_nodes()
                cur_tree.add_node(child_id, smi=child, depth=child_depth)
                cur_tree.add_edge(parent_id, child_id)
                if is_parent[child]:
                    cur_expanded_trees = _recursively_expand_tree(child_id, child, cur_tree)
                else:
                    cur_expanded_trees = [cur_tree.copy()]

                expanded_trees += cur_expanded_trees

        return expanded_trees

    def _construct_trees(root):
        base_tree = nx.DiGraph()
        base_tree.add_node(0, smi=root, depth=0)
        trees = _recursively_expand_tree(0, root, base_tree)
        return trees

    roots = list(filter(lambda smi: not is_child[smi], vocab_smis))
    trees = []
    for root in tqdm(roots):
        new_trees = _construct_trees(root)
        trees.extend(new_trees)

    with open(f"{PROCESSED_DATA_DIR}/synth_trees.pt", "wb") as f:
        torch.save(trees, f)
    # depths = [max(list(nx.get_node_attributes(tree,'depth').values())) for tree in trees]
    # depth_types = set(depths)
    # depth_nums = [len(list(filter(lambda depth: depth==depth_type, depths))) for depth_type in depth_types]
    # print(depth_types)
    # print(depth_nums)
    # print(len(depths))

