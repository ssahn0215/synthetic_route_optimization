def is_valid_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return True
    except:
        return False