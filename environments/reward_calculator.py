import os, sys

from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer


def _get_largest_ring_size(mol):
    cycle_list = mol.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length


def _penalized_logp(mol):
    log_p = Descriptors.MolLogP(mol)
    log_p = (log_p - 2.4729421499641497) / 1.4157879815362406

    sas_score = sascorer.calculateScore(mol)
    sas_score = (sas_score - 3.0470797085649894) / 0.830643172314514

    largest_ring_size = _get_largest_ring_size(mol)
    cycle_score = max(largest_ring_size - 6, 0)
    cycle_score = (cycle_score - 0.038131530820234766) / 0.2240274735210179

    score = log_p - sas_score - cycle_score

    return score

class RewardCalculator():
    def __init__(self):
        self.reward_func = _penalized_logp

    def get_rewards(self, smis):
        mols = [MolFromSmiles(smi) for smi in smis]
        rewards = [self.reward_func(mol) for mol in mols]
        return rewards
    
    def get_reward(self, smi):
        mol = MolFromSmiles(smi)
        reward = self.reward_func(mol)
        return reward