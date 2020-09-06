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
    lr = 1e-3
    batch_size = 128
    num_steps = 100000
    eval_freq = 1000

    dataset = SyntheticTreeDataset()
    num_atom_features = dataset.vocab_graphs[0].ndata["x"].size(1)
    num_bond_features = dataset.vocab_graphs[0].edata["w"].size(1)
    num_vocabs = len(dataset.leaf_smis)
    device = torch.device(0)
    #device = torch.device("cpu")
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
    tree_propagation_model.to(device)

    optimizer = Adam(tree_propagation_model.parameters(), lr=lr)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=synthetic_tree_collate_fn
    )

    criterion = torch.nn.CrossEntropyLoss()
    if not disable_neptune:
        neptune.init(project_qualified_name="sungsoo.ahn/synthetic-route-generation")
        experiment = neptune.create_experiment(name="synthetic-route-generation")

    for step in tqdm(range(num_steps)):
        try:
            graph, node2mol = next(data_iter)
        except:
            data_iter = iter(data_loader)
            graph, node2mol = next(data_iter)

        logits = tree_propagation_model(graph, node2mol, device)
        target = graph.ndata["y"].to(device)
        mask = (target != dataset.mask_target)
        loss = criterion(logits[mask, :], target[mask])

        #loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not disable_neptune:
            neptune.log_metric("loss", loss.item())

        if (step + 1) % eval_freq == 0:
            tree_propagation_model = tree_propagation_model.cpu()
            tree_propagation_model.eval()
            for _ in range(10):
                try:                    
                    graph, node2smi = tree_propagation_model.generate()
                    edges = [(u, v) for u, v in zip(graph.edges()[0].tolist(), graph.edges()[1].tolist())]
                    if not disable_neptune:
                        neptune.log_text("edges", str(edges))
                        neptune.log_text("node2smi", str(node2smi))
                except:
                    neptune.log_text("edges", "generation failed")
                    neptune.log_text("node2smi", "generation failed")
            
            torch.save(tree_propagation_model.state_dict(), f"{CHECKPOINT_DIR}/generator.pt")
            
            tree_propagation_model = tree_propagation_model.to(device)
            tree_propagation_model.train()
