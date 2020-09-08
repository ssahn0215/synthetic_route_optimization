from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph

from data.util import is_valid_smiles

_atom_featurizer = CanonicalAtomFeaturizer(atom_data_field="x")
_bond_featurizer = CanonicalBondFeaturizer(bond_data_field="w")

class MoleculeRepresentations(object):
    def __init__(self, smi, graph):
        self.smi = smi
        self.graph = graph
        
def smi2mol_repr(smi):
    if not is_valid_smiles(smi):
        return None

    graph = smiles_to_bigraph(smi, node_featurizer=_atom_featurizer, edge_featurizer=_bond_featurizer)
    return MoleculeRepresentations(smi, graph)

def smi2molgraph(smi):
    graph = smiles_to_bigraph(smi, node_featurizer=_atom_featurizer, edge_featurizer=_bond_featurizer)
    return graph