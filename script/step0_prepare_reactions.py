import csv, rdkit, pickle
from multiprocessing import Pool
from rdkit import Chem

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from local_config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.RemoveHs(mol)
        for atom in mol.GetAtoms():
            atom.ClearProp("molAtomMapNumber")
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except:
        return ""

def canonicalize_reaction(rxn):
    reactants = canonicalize_smiles(rxn.split(">")[0]).split(".")
    products = canonicalize_smiles(rxn.split(">")[2]).split(".")
    return reactants, products

if __name__ == "__main__":
    assert rdkit.__version__.startswith("2020.03")

    for split in ["train", "val", "test"]:
        with open(f"{RAW_DATA_DIR}/raw_{split}.csv") as f:
            reader = csv.DictReader(f)
            reactions = list(map(lambda row: row["reactants>reagents>production"], reader))

        with Pool(16) as p:
            reactions = p.map(canonicalize_reaction, reactions)

        reactions = list(
            filter(
                lambda rxn: not ("" in rxn[0] or "" in rxn[1] or len(rxn[1]) != 1 or rxn[1][0] in rxn[0]), reactions
            )
        )

        reactions = [(tuple(r), p[0]) for r, p in reactions]
        
        print(f"Number of reactions in {split} split: {len(reactions)}")

        processed_filename = f"{PROCESSED_DATA_DIR}/processed_{split}.pkl"
        with open(processed_filename, 'wb') as f:
            pickle.dump(reactions, f)
