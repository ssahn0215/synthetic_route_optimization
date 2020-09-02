import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import dgl

from dataset import SyntheticTreeDataset, synthetic_tree_collate_fn
from model import SyntheticTreePropagationNetwork
from tqdm import tqdm

import neptune

if __name__ == "__main__":
    num_layers = 2
    num_hidden_features = 128
    lr = 1e-3
    batch_size = 32
    num_steps = 50000

    dataset = SyntheticTreeDataset()    
    num_atom_features = dataset.vocab_graphs[0].ndata['x'].size(1)
    num_bond_features = dataset.vocab_graphs[0].edata['w'].size(1)
    num_vocabs = len(dataset.leaf_smis)

    device = torch.device(0)
    tree_propagation_model = SyntheticTreePropagationNetwork(
        num_vocabs, num_layers, num_hidden_features, num_atom_features, num_bond_features
        )
    tree_propagation_model.to(device)

    optimizer = Adam(tree_propagation_model.parameters(), lr=lr)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=synthetic_tree_collate_fn
        )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    neptune.init(project_qualified_name="sungsoo.ahn/forward-synthesis")
    experiment = neptune.create_experiment(name="forward-synthesis", params=vars(args))
    
    for step in tqdm(range(num_steps)):
        try:
            graph, node2mol = next(data_iter)
        except:
            data_iter = iter(data_loader)
            graph, node2mol = next(data_iter)

        logits = tree_propagation_model(graph, node2mol, device)
        target = graph.ndata["y"].to(device)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        neptune.log_metric("loss", loss.item())