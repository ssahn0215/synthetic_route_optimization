from environments.reaction_predictor import ReactionPredicter
from environments.reward_calculator import RewardCalculator
from environments.binary_synthesis import BinarySynthesisEnvironments

from model.agent import ActorCriticAgent, ActorCriticTrainer

from torch.optim import Adam

import pickle

import torch

from tqdm import tqdm

import argparse

import neptune

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ingredients", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--max_t", type=int, default=3)
    parser.add_argument("--entropy_coef", type=float, default=0.1)
    parser.add_argument("--actor_lr", type=float, default=1e-5)
    parser.add_argument("--critic_lr", type=float, default=1e-5)
    parser.add_argument("--num_episodes", type=int, default=1000)
    args = parser.parse_args()

    reaction_predictor = ReactionPredicter()
    reward_calculator = RewardCalculator()
    with open("./resource/schneider50k/processed.pkl", "rb") as f:
        ingredient_smis = list(pickle.load(f))[:args.num_ingredients]

    env = BinarySynthesisEnvironments(reaction_predictor, reward_calculator, ingredient_smis, args.num_envs, args.max_t)

    entropy_coef = args.entropy_coef
    device = torch.device(0)
    agent = ActorCriticAgent(device=device)
    actor_optimizer = Adam(agent.actor_net.parameters(), lr=args.actor_lr)
    critic_optimizer =  Adam(agent.critic_net.parameters(), lr=args.critic_lr)
    trainer = ActorCriticTrainer(agent, actor_optimizer, critic_optimizer, entropy_coef, device)

    neptune.init(project_qualified_name="sungsoo.ahn/forward-synthesis")
    experiment = neptune.create_experiment(name="forward-synthesis", params=vars(args))

    max_reward = 0.0
    for _ in tqdm(range(args.num_episodes)):
        episode_values, episode_log_probs, episode_rewards, episode_dones, episode_entropies = [], [], [], [], []
        obs, dones = env.init()
        while torch.any(~dones):
            actions, values, log_probs, entropies = agent.act_and_crit(obs, dones)
            
            obs, rewards, dones, info = env.step(actions)
            
            episode_values.append(values)
            episode_log_probs.append(log_probs)
            episode_entropies.append(entropies)
            episode_rewards.append(rewards)
            episode_dones.append(dones)
        
        actor_loss, critic_loss = trainer.train(
            episode_values, episode_log_probs, episode_entropies, episode_rewards, episode_dones
            )

        neptune.log_metric("actor_loss", actor_loss.item())
        neptune.log_metric("critic_loss", critic_loss.item())
        
        
        avg_reward = torch.stack(episode_rewards, dim=0).max(dim=0)[0].mean(dim=0).item()
        max_reward = max(max_reward, torch.cat(episode_rewards).max().item())

        neptune.log_metric("max_reward", max_reward)
        neptune.log_metric("avg_reward", avg_reward)
