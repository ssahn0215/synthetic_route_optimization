import os

from rdkit.Chem import Descriptors, MolToSmiles, AllChem, MolFromSmiles

import torch

current_path = os.path.dirname(os.path.abspath(__file__))
descriptors_list = []
with open("./resource/descriptors.txt", "r") as f:
    for line in f:
        descriptors_list.append(line.strip())

descriptors_dict = dict(Descriptors.descList)


def featurize_molecule(mol, features):
    features_list = []
    for feature in features:
        features_list.extend(feat_dict[feature['type']](mol, feature))
    return features_list


def ecfp(molecule, options):
    return [x for x in AllChem.GetMorganFingerprintAsBitVect(
        molecule, options['radius'], options['length'])]


def rdkit_headers():
    headers = [x[0] for x in Descriptors.descList]
    return headers


def fingerprint_headers(options):
    return ['{}{}_{}'.format(options['type'], options['radius'], x) for x in range(options['length'])]


def rdkit_descriptors(molecule, options=None):
    descriptors = []
    for desc_name in descriptors_list:
        try:
            desc = descriptors_dict[desc_name]
            bin_value = desc(molecule)
        except (ValueError, TypeError, ZeroDivisionError) as exception:
            print(
                'Calculation of the Descriptor {} failed for a molecule {} due to {}'.format(
                    str(desc_name), str(MolToSmiles(molecule)), str(exception))
            )
            bin_value = 'NaN'

        descriptors.append(bin_value)

    return descriptors


# TODO: add option to parallelize
class MolculeEmbedder(object):
    def __init__(self):
        self.embedding_func = None
        self.default_embedding = None

    def get_embeddings(self, smis):
        return torch.stack([self._compute_embedding(smi) for smi in smis], dim=0)

    def get_embedding(self, smi):
        return self._compute_embedding(smi)

    def _compute_embedding(self, smi):
        if smi == "":
            return self.default_embedding
        else:
            mol = MolFromSmiles(smi)
            descriptors = self.embedding_func(mol)
            embeddings = torch.tensor(descriptors)
            embeddings[embeddings!=embeddings] = 0.0 
            return embeddings

class RdkitDescriptorMoleculeEmbedder(MolculeEmbedder):
    def __init__(self):
        self.embedding_func = rdkit_descriptors
        self.default_embedding = torch.zeros(len(descriptors_list))
        self.embedding_dim = len(descriptors_list)

class ECFPMoleculeEmbedder(MolculeEmbedder):
    def __init__(self):
        self.embedding_func = lambda mol: ecfp(mol, options={"radius": 2, "length": 1024})
        self.default_embedding = torch.zeros(1024)
        self.embedding_dim = 1024

if __name__ == "__main__":
    embedder = ECFPMoleculeEmbedder()
    smis = "C1CCOC1.N#Cc1ccsc1N.O=[N+]([O-])c1cc(F)c(F)cc1F".split(".")
    print(embedder.get_embeddings(smis))