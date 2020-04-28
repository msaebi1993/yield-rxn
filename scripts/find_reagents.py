        curr_rct = set(rct_smiles[i])

strings = []
bond_changes = []
for line in open("./data/test.txt.proc", "r"):
    string = line.split()[0]
    bond_change = line.split()[1]
    strings.append(string)
    bond_changes.append(bond_change)

new_strings = []
for string in strings:
    rct_string, reag_string, prod_string = string.split(">")
    rct_mol_strings = rct_string.split(".")
    prod_mol_strings = prod_string.split(".")
    rct_mols = [Chem.MolFromSmiles(mol_string) for mol_string in rct_mol_strings]
    prod_mols = [Chem.MolFromSmiles(mol_string) for mol_string in prod_mol_strings]
    rct_smiles = [Chem.MolToSmiles(mol) for mol in rct_mols]
    prod_smiles = [Chem.MolToSmiles(mol) for mol in prod_mols]
    new_reags = []
    new_rcts = []
    for i, mol in enumerate(rct_mols):
        in_prods = False
        curr_rct = set(rct_smiles[i])
        for j, mol in enumerate(prod_mols):
            curr_prod = set(prod_smiles[j])
            if (curr_rct == curr_prod):
                in_prods = True
                break           
        if in_prods:
            new_rcts.append(rct_mol_strings[i])
        else:
            new_reags.append(rct_mol_strings[i])
    new_rct_string = ".".join(new_rcts)
    new_reag_string = ".".join(new_reags)
    new_string = ">".join([new_rct_string, new_reag_string, prod_string])
    new_strings.append(new_string)

with open("./data/tmp_reag.txt.proc", "w") as f:
    for string, edits in zip(new_strings, bond_changes):
        try:
            f.write(" ".join([string, edits]))
            f.write("\n")
        except Exception as ex:
            print(ex)
            print(string)
            print(edits)
