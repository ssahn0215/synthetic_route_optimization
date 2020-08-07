import torch
import torch.nn as nn
from model.molecule_embedder import RdkitDescriptorMoleculeEmbedder
from torch.distributions import Categorical, Bernoulli

class ActorNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(ActorNetwork, self).__init__()
        self.linear0 = nn.Linear(embedding_dim, 2048)
        self.relu0 = nn.ReLU()
        self.linear1 = nn.Linear(2048, embedding_dim+1)
    
    def forward(self, x):
        out = self.linear0(x)
        out = self.relu0(out)
        out = self.linear1(out)

        return out

class CriticNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(CriticNetwork, self).__init__()
        self.linear0 = nn.Linear(embedding_dim, 2048)
        self.relu0 = nn.ReLU()
        self.linear1 = nn.Linear(2048, 1)
    
    def forward(self, x):
        out = self.linear0(x)
        out = self.relu0(out)
        out = self.linear1(out)

        return out.squeeze(1)


class ActorCriticAgent(nn.Module):
    def __init__(self, device):
        super(ActorCriticAgent, self).__init__()
        self.mol_embedder = RdkitDescriptorMoleculeEmbedder()
        self.embedding_dim = self.mol_embedder.embedding_dim
        self.actor_net = ActorNetwork(self.embedding_dim)
        self.critic_net = CriticNetwork(self.embedding_dim)
        self.device = device
        self.to(device)

    def act_and_crit(self, obs, dones):
        current_smis = [] 
        for state_smi, done in zip(obs.state_smis, dones.tolist()):
            if not done:
                current_smis.append(state_smi)
        
        ingredient_smis = obs.ingredient_smis

        current_embeddings = self.mol_embedder.get_embeddings(current_smis)
        current_embeddings = current_embeddings.float().to(self.device)

        ingredient_embeddings = self.mol_embedder.get_embeddings(ingredient_smis)            
        ingredient_embeddings = ingredient_embeddings.float().to(self.device)

        actor_out = self.actor_net(current_embeddings)
        actor_embeddings, halt_logits = torch.split(actor_out, [self.embedding_dim, 1], dim=1)
        select_logits = torch.matmul(actor_embeddings, torch.t(ingredient_embeddings)) / ingredient_embeddings.size(1)

        select_entropies = - torch.sum(
            torch.softmax(select_logits, dim=1) *  torch.log_softmax(select_logits, dim=1), dim=1
            )

        select_distrib = Categorical(logits=select_logits)

        selects = select_distrib.sample()
        select_log_probs = select_distrib.log_prob(selects)            
        if obs.t > 0:
            halt_distrib = Bernoulli(torch.sigmoid(halt_logits.squeeze(1)))
            halts = halt_distrib.sample()            
            halt_log_probs = halt_distrib.log_prob(halts)

            actions = selects.clone()
            actions[halts == 1] = len(ingredient_smis) # somehow change to HALT_ACTION
            log_probs = (halts == 0).float() * select_log_probs + halt_log_probs

        else:
            actions = selects.clone()
            log_probs = select_log_probs

        padded_log_probs = torch.zeros_like(dones).float().to(self.device)
        padded_log_probs[~dones] = log_probs

        values = self.critic_net(current_embeddings)
        padded_values = torch.zeros_like(dones).float().to(self.device)
        padded_values[~dones] = values

        
        return actions, padded_values, padded_log_probs, select_entropies


class ActorCriticTrainer(nn.Module):
    def __init__(self, agent, actor_optimizer, critic_optimizer, entropy_coef, device):
        super(ActorCriticTrainer, self).__init__()
        self.agent = agent
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.entropy_coef = entropy_coef
        self.device = device

    def train(self, values, log_probs, entropies, rewards, dones):
        values = torch.stack(values, dim=0).to(self.device)
        log_probs = torch.stack(log_probs, dim=0).to(self.device)
        rewards = torch.stack(rewards, dim=0).to(self.device)
        dones = torch.stack(dones, dim=0).to(self.device)
        
        returns = self.compute_returns(rewards, dones).detach()
        advantage = returns - values

        mean_entropy = torch.sum(torch.stack([entropies_.sum() for entropies_ in entropies])) / torch.sum(~dones).float()

        actor_loss = -(log_probs * advantage.detach()).mean() - self.entropy_coef * mean_entropy
        critic_loss = advantage.pow(2).mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss.detach(), critic_loss.detach()

    def compute_returns(self, rewards, dones):
        R = 0.0
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + R * dones[step].float()
            returns.insert(0, R)

        return torch.stack(returns, dim=0)