import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import dgl

from dataset import SyntheticTreeDataset, synthetic_tree_collate_fn
from model import SyntheticTreePropagationNetwork
from reaction_predictor import ReactionPredicter
from local_config import CHECKPOINT_DIR

from tqdm import tqdm
import neptune

if __name__ == "__main__":
    num_layers = 5
    num_hidden_features = 1024
    dataset = SyntheticTreeDataset()
    num_atom_features = dataset.vocab_graphs[0].ndata["x"].size(1)
    num_bond_features = dataset.vocab_graphs[0].edata["w"].size(1)
    num_vocabs = len(dataset.leaf_smis)
    disable_neptune = False

    reaction_predictor = ReactionPredicter()
    tree_propagation_model = SyntheticTreePropagationNetwork(
        num_vocabs,
        num_layers,
        num_hidden_features,
        num_atom_features,
        num_bond_features,
        reaction_predictor,
        dataset.leaf_smis,
        dataset.leaf_graphs,
    )

    state_dict = torch.load(f"{CHECKPOINT_DIR}/generator.pt")
    tree_propagation_model.load_state_dict(state_dict)
    tree_propagation_model.eval()        
    for _ in range(100):
        graphs, node2smis, node2graphs = tree_propagation_model._generate()
        print(node2smis)
