import torch
from torch.optim import Adam
import dgl
from tqdm import tqdm
import neptune
import argparse

from data.dataset import SyntheticRouteDataset, synthetic_route_collate_fn
from data.synthetic_route import load_building_block_data
from module.model import SyntheticRouteNetwork
from runner.generator import SyntheticRouteGenerator
from runner.distribution_learning_trainer import DistributionLearningTrainer
from local_config import NEPTUNE_ID, NEPTUNE_PROJECT_NAME, CHECKPOINT_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_embedding_layers', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--generate_batch_size', type=int, default=32)
    parser.add_argument('--num_eval_samples', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=500000)
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--checkpoint_freq', type=int, default=1000)
    parser.add_argument('--disable_neptune', action="store_true")
    parser.add_argument('--disable_cuda', action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.disable_cuda else torch.device(0)

    dataset = SyntheticRouteDataset()
    building_block_smis, building_block_molgraphs = load_building_block_data()
    input_node_dim = building_block_molgraphs[0].ndata["x"].size(1)
    input_edge_dim = building_block_molgraphs[0].edata["w"].size(1)
    num_building_blocks = len(building_block_smis)

    device = torch.device(device)

    model = SyntheticRouteNetwork(
        num_embedding_layers=args.num_embedding_layers,
        hidden_dim=args.hidden_dim, 
        input_node_dim=input_node_dim, 
        input_edge_dim=input_edge_dim, 
        output_dim=num_building_blocks+2
        )
    model = model.to(device)    
    optimizer = Adam(model.parameters(), lr=args.lr)

    generator = SyntheticRouteGenerator(
        model=model,
        building_block_smis=building_block_smis,
        building_block_molgraphs=building_block_molgraphs,
        batch_size=args.generate_batch_size,
        device=device,
    )

    if not args.disable_neptune:
        neptune.init(project_qualified_name=f"{NEPTUNE_ID}/{NEPTUNE_PROJECT_NAME}")
        experiment = neptune.create_experiment(name="distribution-learning", params=vars(args))
        checkpoint_path = f"{CHECKPOINT_DIR}/distribution_learning/{experiment._id}.pt"
    else:
        checkpoint_path = f"{CHECKPOINT_DIR}/distribution_learning/tmp.pt"

    trainer = DistributionLearningTrainer(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        num_steps=args.num_steps,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        batch_size=args.train_batch_size,
        num_eval_samples=args.num_eval_samples, 
        checkpoint_path=checkpoint_path,
        generator=generator,
        device=device,
        disable_neptune=args.disable_neptune,
    )


    trainer.train()