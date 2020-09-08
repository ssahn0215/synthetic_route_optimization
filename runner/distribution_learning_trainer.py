from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import dgl
import neptune

from data.dataset import synthetic_route_collate_fn
from data.synthetic_route import PARENT_EDGE_TYPE


class DistributionLearningTrainer:
    def __init__(
        self,
        model,
        optimizer,
        dataset,
        num_steps,
        eval_freq,
        checkpoint_freq,
        batch_size,
        num_eval_samples,
        checkpoint_path,
        generator,
        device,
        disable_neptune,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.num_steps = num_steps
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.batch_size = batch_size
        self.num_eval_samples = num_eval_samples
        self.checkpoint_path = checkpoint_path
        self.generator = generator
        self.device = device
        self.disable_neptune = disable_neptune
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        data_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=synthetic_route_collate_fn
        )
        for step in tqdm(range(self.num_steps)):
            try:
                graph, node2smi, node2molgraph, prop_traj = next(data_iter)
            except:
                data_iter = iter(data_loader)
                graph, node2smi, node2molgraph, prop_traj = next(data_iter)

            loss = self.train_batch(graph, node2molgraph, prop_traj)
            if not self.disable_neptune:
                neptune.log_metric("loss", loss.item())

            if (step + 1) % self.eval_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    graphs, node2smis, node2molgraphs = self.generator.generate(num_samples=self.num_eval_samples)

                self.model.train()
                if not self.disable_neptune:
                    for graph, node2smi in zip(graphs, node2smis):
                        edges = [
                            (u, v)
                            for u, v in zip(
                                graph.edges(etype=PARENT_EDGE_TYPE)[0].tolist(),
                                graph.edges(etype=PARENT_EDGE_TYPE)[1].tolist(),
                            )
                        ]
                        neptune.log_text("edges", str(edges))
                        neptune.log_text("node2smi", str(node2smi))

            if (step + 1) % self.checkpoint_freq == 0:
                state_dict = self.model.state_dict()
                with open(self.checkpoint_path, "wb") as f:
                    torch.save(state_dict, f)

    def train_batch(self, graph, node2molgraph, prop_traj):
        nonstop_nodes = list(sorted(node2molgraph.keys()))
        molgraphs = [node2molgraph[node] for node in nonstop_nodes]
        batch_mol_graph = dgl.batch(molgraphs).to(self.device)
        mol_z = self.model.embedding_layer(batch_mol_graph)

        graph = graph.to(self.device)
        num_nodes = graph.number_of_nodes()

        graph.ndata["mol_z"] = self.model.get_default_mol_z(num_nodes).to(self.device)
        graph.ndata["mol_z"][nonstop_nodes] = mol_z
        graph.ndata["prop_z"] = self.model.get_default_prop_z(num_nodes).to(self.device)

        for prop_nodes in prop_traj:
            graph = self.model.propagation_layer(graph, prop_nodes)

        graph.ndata.pop("mol_z")
        prop_z = graph.ndata.pop("prop_z")
        logits = self.model.classification_layer(prop_z)

        target = graph.ndata["target"]
        mask = target < self.model.output_dim
        loss = self.criterion(logits[mask, :], target[mask])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
