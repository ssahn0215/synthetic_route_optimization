import os
import csv
from tqdm import tqdm

from config import RAW_USPTO50K_DIR
from common.mol_utils import cano_smiles, cano_smarts, smarts_has_useless_parentheses
from external.rdchiral.template_extractor import extract_from_reaction

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from external.rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun

if __name__ == "__main__":
    uspto50k_reactant_smiles = []
    uspto50k_template_smarts = []
    uspto50k_rxns = []

    for split in ["train", "val", "test"]:
        raw_uspto50k_filename = os.path.join(RAW_USPTO50K_DIR, f"raw_{split}.csv")
        with open(raw_uspto50k_filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rxn_idx = header.index('reactants>reagents>production')
            for row_idx, row in list(enumerate(reader)):
                reactants, reagents, product = row[rxn_idx].split(">")

                reactant_mols, reactant_smiless = map(list, zip(*map(cano_smiles, reactants.split("."))))
                product_mol, product_smiles = cano_smiles(product)
                if any(mol is None for mol in reactant_mols + [product_mol]):
                    assert False                
                
                #uspto50k_reactant_smiles.extend(reactant_smiless)
                
                rxn = {'_id': row_idx, 'reactants': reactants, 'products': product}
                template = extract_from_reaction(rxn).get('reaction_smarts', None)
                if template is None:
                    assert False

                #product_smarts, _, reactant_smartss = template.split('>')
                #template = '{}>>{}'.format(reactant_smartss, product_smarts)

                try:
                    #rxn = rdChemReactions.ReactionFromSmarts(template)
                    #product_mol = Chem.MolFromSmiles(product_smiles)
                    #result = rxn.RunReactants([product_mol])
                    #reactant_smiles_ = [tuple(sorted(map(Chem.MolToSmiles, r))) for r in result]
                    #assert tuple(sorted(reactant_smiless)) in reactant_smiles_

                    rxn = rdchiralReaction(template)
                    src = rdchiralReactants(product_smiles)
                    result = rdchiralRun(rxn, src)

                except:
                    print(row_idx)
                    print(template)
                    print(product_smiles)
                    print(result)
                    print(reactant_smiless)
                    print("="*10)
                    assert False
