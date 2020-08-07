import os
import csv
from config import RAW_USPTO50K_DIR, PROCESSED_USPTO50K_DIR
from rdkit import Chem
from tqdm import tqdm
import pickle
def canonicalize(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
        if tmp is None:
            return None, smiles        
        tmp = Chem.RemoveHs(tmp)
        if tmp is None:
            return None, smiles
        [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
        return tmp, Chem.MolToSmiles(tmp)            
    except:
        return None, smiles

if __name__ == "__main__":
    uspto50k_reactant_smis = set()
    for split in ["train", "val", "test"]:
        raw_uspto50k_filename = os.path.join(RAW_USPTO50K_DIR, f"raw_{split}.csv")
        with open(raw_uspto50k_filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            reaction_idx = header.index('reactants>reagents>production')
            for row_idx, row in tqdm(list(enumerate(reader))):
                reactants, reagents, product = row[reaction_idx].split(">")

                for reactant in reactants.split("."):
                    mol, smi = canonicalize(reactant)
                    if mol is not None:
                        uspto50k_reactant_smis.add(smi)
    
    processed_uspto50k_filename = os.path.join(PROCESSED_USPTO50K_DIR, f"processed.pkl")
    with open(processed_uspto50k_filename, 'wb') as f:
        pickle.dump(uspto50k_reactant_smis, f)