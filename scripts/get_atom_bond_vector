def get_A_B_Y_Matrix_from_sdf(CID)
                               # Read the SDF files based on the CID of the molecule and get atom index vector bond binary type matrix and bond type 
                                 
    a_index=None  #atom index
    bond_matrix=None   #bond binary type matrix
    bd=None  #bond type featrue vector
    try:
        os.chdir('../SDF_')
        atom_index = []
        a_index = 1
        m = Chem.MolFromMolFile('{}.sdf'.format(CID))
        for atom in m.GetAtoms():
            atom_index.append(a_index)
            a_index = a_index + 1
        BD = []
        for bond in m.GetBonds():
            bd = [0, 0, 0]
            if str(bond.GetBondType()) == 'SINGLE':
                bd[0] = 1
            elif str(bond.GetBondType()) == 'AROMATIC':
                bd[1] = 1
            else:
                bd[2] = 1
            BD.append(bd)
        Num_atom = m.GetNumAtoms()
        bond_matrix = np.zeros([Num_atom, Num_atom])
        for i in range(Num_atom):
            for j in range(Num_atom):
                if m.GetBondBetweenAtoms(i, j) == None:
                    bond_matrix[i, j] = 0
                else:
                    bond_matrix[i, j] = 1
    except:
        return 0
    return atom_index, bond_matrix, bd
    
    
    
os.chdir('../json')
with open('1_1000.json') as f:
    chemdict=json.load(f)
    for reaction in chemdict['Reactions']:
        reactants=reaction['reactants']
        for reactant in reactants:
            CID = reactant['CID']
            atoms = reactant['atoms']
            atom_matrix=[]        #atom feature vector
            for atom in atoms:
                atomic_num = atom['atomic_num']
                nmr_shift = atom['nmr_shift']
                partial_charge = atom['partial_charge']
                at=[atomic_num,nmr_shift,partial_charge]
                atom_matrix.append(at)
            if type(CID)==str and CID in list:
                A,B,Y=get_A_B_Y_Matrix_from_sdf(CID)
            reactant_map=(A,B,atom_matrix,Y)
            print(reactant_map)
