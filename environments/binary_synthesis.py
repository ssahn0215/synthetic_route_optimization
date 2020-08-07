from onmt.translate.silent_translator import build_default_translator
import re
import torch
from rdkit import Chem

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

# TODO: change states/observations/info to dataclasses
# from dataclasses import dataclass


def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens)
    return " ".join(tokens)


class BinarySynthesisStates:
    def __init__(self, smis, dones):
        self.smis = smis
        self.dones = dones


class BinarySynthesisObservations:
    def __init__(self, state_smis, ingredient_smis, t):
        self.state_smis = state_smis
        self.ingredient_smis = ingredient_smis
        self.t = t

class BinarySynthesisInfo:
    def __init__(self, reaction_scores):
        self.reaction_scores = reaction_scores


class BinarySynthesisEnvironments(object):
    def __init__(self, reaction_predictor, reward_calculator, ingredient_smis, num_envs, max_t):
        self.num_envs = num_envs
        self.reaction_predictor = reaction_predictor
        self.reward_calculator = reward_calculator
        self.ingredient_smis = ingredient_smis
        self.HALT_ACTION = len(self.ingredient_smis)
        self.max_t = max_t

    def init(self):
        smis = ["" for _ in range(self.num_envs)]
        dones = torch.tensor([False for _ in range(self.num_envs)])
        self.states = BinarySynthesisStates(smis=smis, dones=dones)
        self.t = 0

        obs = BinarySynthesisObservations(state_smis=self.states.smis, ingredient_smis=self.ingredient_smis, t=self.t)
        
        return obs, dones

    def step(self, actions):
        actions = actions.cpu()
        reaction_scores = None
        if self.t == 0:
            active_action_smis = [self.ingredient_smis[action] for action in actions]
            self.states.smis = active_action_smis
            rewards = torch.zeros(len(actions))
        else:
            active_state_idxs = (~self.states.dones).nonzero().squeeze(1).tolist()
            active_state_idxs_and_actions = [
                        (active_state_idxs[idx], action)
                        for (idx, action) in enumerate(actions)
                        if action != self.HALT_ACTION
                    ]

            failed_state_idxs = []
            if len(active_state_idxs_and_actions) > 0:
                active_state_idxs, active_actions = map(list, zip(*active_state_idxs_and_actions))
                active_state_smis = [self.states.smis[idx] for idx in active_state_idxs]
                active_action_smis = [self.ingredient_smis[action] for action in active_actions]

                reactant_sets = [
                    [reactant0, reactant1] for reactant0, reactant1 in zip(active_state_smis, active_action_smis)
                ]
                reaction_scores, products = self.reaction_predictor.get_products(reactant_sets)

                for product_idx, state_idx in enumerate(active_state_idxs):
                    product = products[product_idx]
                    if self.is_valid(product):
                        self.states.smis[state_idx] = product
                    else:
                        self.states.smis[state_idx] = ""
                        failed_state_idxs.append(state_idx)

            
            active_state_idxs = (~self.states.dones).nonzero().squeeze(1).tolist()
            rewards = torch.zeros(self.num_envs)
            for idx, action in enumerate(actions):
                if action == self.HALT_ACTION or self.t == self.max_t:
                    smi = self.states.smis[active_state_idxs[idx]]
                    if smi != "":
                        reward = self.reward_calculator.get_reward(smi)
                        rewards[idx] = reward

            self.states.dones[~self.states.dones] = (actions == self.HALT_ACTION)
            self.states.dones[~self.states.dones] = (self.t == self.max_t)
            self.states.dones[failed_state_idxs] = True

        self.t += 1

        obs = BinarySynthesisObservations(state_smis=self.states.smis, ingredient_smis=self.ingredient_smis, t=self.t)
        info = BinarySynthesisInfo(reaction_scores=reaction_scores)
        dones = self.states.dones.clone()

        return obs, rewards, dones, info

    def is_valid(self, smi):
        mol = Chem.MolFromSmiles(smi)
        return (mol is not None)