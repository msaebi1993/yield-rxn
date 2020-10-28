from __future__ import print_function
from __future__ import division

import os
import logging
import rdkit.Chem as Chem
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
import json
from collections import defaultdict

from .reaction import Rxn

class RxnDataset(Dataset):
    """Object for containing sets of reactions SMILES strings.

    Attributes:
        file      (str): location of the file rxns are loaded from.
        rxn_strs  (set): reaction strings of the dataset and the bonding changes.
        bins     (dict): contains lists of indices to map rxn_strs to bin sizes
                for mapping data into efficient batches.

    """
    def __init__(self, file_name, path="data/", vocab=None):
        super(RxnDataset, self).__init__()
        self.file_name = file_name
        self.path = path
        self.vocab = vocab

    def __len__(self):
        return len(self.rxns)

    def __setitem__(self, idx, value):
        self.rxns[idx] = value

    @property
    def rxn_smiles(self):
        return [rxn.smile for rxn in self.rxns]

    @rxn_smiles.setter
    def rxn_smiles(self, rxn_smiles):
        self.rxns = [Rxn(rxn_smile) for rxn_smile in rxn_smiles]

    @classmethod
    def from_list(cls, rxns):
        new_dataset = cls('')
        new_dataset.rxns = rxns
        return new_dataset

    def save_to_file(self, file_name=None, path=None):
        if file_name == None:
            file_name = self.file_name
        if path == None:
            path = self.path
        with open(os.path.join(path, file_name), "w") as f:
            for rxn in self.rxn_smiles:
                f.write(rxn+"\n")

    def canonicalize(self):
        for rxn in self.rxns:
            rxn.canonicalize()

    def remove_rxn_mappings(self):
        for rxn in self.rxns:
            rxn.remove_rxn_mapping()

    def remove_max_reactants(self, max_reactants):
        keep_rxns = [rxn for rxn in self.rxns if len(rxn.reactants) <= max_reactants]
        self.rxns = keep_rxns

    def remove_max_products(self, max_products):
        keep_rxns = [rxn for rxn in self.rxns if len(rxn.products) <= max_products]
        self.rxns = keep_rxns


        
class RxnGraphDataset(RxnDataset):
    """Object for containing sets of reactions SMILES strings.

    Attributes:
        file      (str): location of the file rxns are loaded from.
        rxn_strs  (set): reaction strings of the dataset and the bonding changes.
        bins     (dict): contains lists of indices to map rxn_strs to bin sizes
                for mapping data into efficient batches.
    """    
    def __init__(self, file_name, path):
        super(RxnGraphDataset, self).__init__(file_name, path)
        self.file_name = file_name
        self.path = path
        self.rxns = []
        self.degree_codec = LabelEncoder()
        self.symbol_codec = LabelEncoder()
        self.expl_val_codec = LabelEncoder()
        self.bond_type_codec = LabelEncoder()
        self._init_dataset()
        self.max_nbonds = 10

    def _init_dataset(self):
        logging.info("Loading Dataset {dataset} in {datapath}".format(
            dataset=self.file_name, datapath=self.path))
        symbols = set()
        degrees = set()
        explicit_valences = set()
        bond_types = set()
        
        with open(os.path.join(self.path, self.file_name)) as datafile:
            data = json.load(datafile)
            for line in data:
                product=line['product']
                reactants=line['reactants']
                r_yield=line['yield']
     
                rxn = Rxn(product,reactants,r_yield)
                self.rxns.append(rxn)
                mol = Chem.MolFromSmiles(rxn.reactants_smile)

                for atom in mol.GetAtoms():   
                    symbols.add(atom.GetSymbol())
                    degrees.add(atom.GetDegree())
                    explicit_valences.add(atom.GetExplicitValence())
                for bond in mol.GetBonds():
                    bond_types.add(bond.GetBondType())
            
                    
        symbols.add("unknown")
        logging.info("Dataset contains {:d} total samples".format(len(self.rxns)))

        self.degree_codec.fit(list(degrees))
        self.symbol_codec.fit(list(symbols))
        self.expl_val_codec.fit(list(explicit_valences))
        self.bond_type_codec.fit(list(bond_types))
        
    def __getitem__(self, idx):
        rxn = self.rxns[idx]
        react_smiles = rxn.reactants_smile
        
        rxn_feats,mod_idx_feats=self.get_reaction_features(rxn)
        return rxn_feats

    
    max_num_reactants=4
    def get_reaction_features(self,rxn): #rxn is a Rxn object
        reactants=rxn.reactants # is a mol_obj
        #rxn_feats=self.get_molecule_features(reactants[0])#initialized with the first reactant
        rxn_feats=defaultdict(lambda:defaultdict())

        domain_feats=reactants[0].get_attributes()
        mol_idx_feat=defaultdict(lambda:defaultdict())
        
        rxn_feats=self.get_molecule_features(mol_obj=None,smiles=rxn.reactants_smile)
        rxn_feats["yield_label"]=torch.tensor([rxn.yields])
        #print(rxn_feats["yield_label"])
        #for mol_idx in range(1,len(reactants)):
            #domain_feats=torch.cat((domain_feats,reactants[mol_idx].get_attributes()),dim=0) #retunrs attributes from chemists
        
        #rxn_feats["domain_feats"]=domain_feats
        
        #for mol_idx in range(len(reactants)):
            #mol_feat=self.get_molecule_features(reactants[mol_idx])

            #for feat in mol_feat.keys():
                #mol_idx_feat[feat][mol_idx]=mol_feat
                #if mol_idx>=1:
                    #print("\t feat name: ",feat)
                    #print("concatenated feats: ",output_feats[feat].shape)
                    #print("mold feats: ",mol_feat[feat].shape,"\n")
                    #rxn_feats[feat]=torch.cat((rxn_feats[feat],mol_feat[feat]),dim=0) 
        return rxn_feats,mol_idx_feat


    def get_molecule_features(self,mol_obj=None,smiles=None):
        output=defaultdict(lambda:defaultdict())
        if smiles!=None:
            mol=Chem.MolFromSmiles(smiles)
        else:
            
            smiles=mol_obj.smile
            mol=Chem.MolFromSmiles(mol_obj.smile)
        atom_idx = torch.tensor([atom.GetIdx()-1 for atom in mol.GetAtoms()], dtype=torch.int64)


        n_atoms = mol.GetNumAtoms()
        sparse_idx = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                sparse_idx.append([atom_idx[i],atom_idx[j]])
        sparse_idx = torch.tensor(sparse_idx, dtype=torch.int64)
        atom_feats = self.get_atom_features(mol)
        bond_feats = self.get_bond_features(mol)
        atom_graph = torch.zeros((n_atoms,self.max_nbonds), dtype=torch.int64)
        bond_graph = torch.zeros((n_atoms, self.max_nbonds), dtype=torch.int64)
        n_bonds = torch.zeros((n_atoms,), dtype=torch.int32)

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()-1
            a2 = bond.GetEndAtom().GetIdx()-1
            idx = bond.GetIdx()
            if n_bonds[a1] == self.max_nbonds or n_bonds[a2] == self.max_nbonds:
                raise Exception(smiles)
            atom_graph[a1, n_bonds[a1]] = a2
            atom_graph[a2, n_bonds[a2]] = a1
            bond_graph[a1, n_bonds[a1]] = idx
            bond_graph[a2, n_bonds[a2]] = idx
            n_bonds[a1] += 1
            n_bonds[a2] += 1

        #bond_labels = get_bond_labels(edits, n_atoms, sparse_idx)
        binary_feats = self.get_binary_features(smiles)
        #domain_features=mol_obj.get_attributes() #retunrs attributes from chemists

        output["atom_feats"]=atom_feats
        output["bond_feats"]=bond_feats
        output["atom_graph"]=atom_graph
        output["bond_graph"]=bond_graph
        output["n_bonds"]=n_bonds
        output["n_atoms"]=n_atoms#torch.tensor([n_atoms])
        #output["bond_labels"][mol_idx]=bond_labels
        output["binary_feats"]=binary_feats
        output["sparse_idx"]=sparse_idx
        #output["domain_feats"]=domain_features
        
        return output    

    #def get_atom_features(self,mol_obj):
    def get_atom_features(self,mol):
        #mol=Chem.MolFromSmiles(mol_obj.smile)
        atom_idx = torch.tensor([atom.GetIdx()-1 for atom in mol.GetAtoms()], dtype=torch.int64)

        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
        expl_vals = [atom.GetExplicitValence() for atom in mol.GetAtoms()]

        t_symbol = self.to_one_hot(self.symbol_codec, symbols)
        t_degree = self.to_one_hot(self.degree_codec, degrees)
        t_expl_val = self.to_one_hot(self.expl_val_codec, expl_vals)
        t_aromatic = torch.tensor([atom.GetIsAromatic() for atom in mol.GetAtoms()]).float().unsqueeze(1)

        #nmr_shifts=torch.tensor([mol_obj.atoms[idx]['nmr_shift'] for idx in range(len(mol_obj.atoms))]).float().unsqueeze(1)
        #partial_charges=torch.tensor([mol_obj.atoms[idx]['partial_charge'] for idx in range(len(mol_obj.atoms))]).float().unsqueeze(1)
        #print(nmr_shifts.shape)
        #print(partial_charges.shape)
        general_feats=torch.cat((t_symbol, t_degree, t_expl_val, t_aromatic), dim=1)[atom_idx]
        #final_fetas=torch.cat((general_feats,torch.tensor([nmr_shifts,partial_charges])),dim=0)
        return general_feats

    def get_bond_features(self,mol):
        bond_types = [bond.GetBondType() for bond in mol.GetBonds()]
        t_bond_types = self.to_one_hot(self.bond_type_codec, bond_types)
        t_conjugated = torch.tensor([bond.GetIsConjugated() for bond in mol.GetBonds()]).float().unsqueeze(1)
        t_in_ring = torch.tensor([bond.IsInRing() for bond in mol.GetBonds()]).float().unsqueeze(1)
        return torch.cat((t_bond_types, t_conjugated, t_in_ring), dim=1)            
    
    def get_binary_features(self,smiles):

        comp = {}
        rxn_mol=Chem.MolFromSmiles(smiles)
        symbols = [atom.GetSymbol() for atom in rxn_mol.GetAtoms()]
        all_atoms=set(symbols)
        n_atoms=len(all_atoms)
        #print(n_atoms)
        max_n_atoms=10
        mapping=dict(zip( all_atoms, range(len(all_atoms))))

        for i, s in enumerate(smiles.split('.')):
            mol = Chem.MolFromSmiles(s)
            for atom in mol.GetAtoms():
                comp[mapping[atom.GetSymbol()]] = i

        n_comp = len(smiles.split('.'))
        rmol = Chem.MolFromSmiles(smiles)
        bond_map = {}
        for bond in rmol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()- 1
            a2 = bond.GetEndAtom().GetIdx() - 1
            bond_map[(a1, a2)] = bond

        binary_feats = torch.zeros((10,max_n_atoms, max_n_atoms))
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if i == j:
                    continue
                if (i,j) in bond_map:
                    bond = bond_map[(i,j)]
                    binary_feats[1:1+6,i,j] = binary_feats[1:1+6,j,i] = self.bond_features(bond)
                elif (j,i) in bond_map:
                    bond = bond_map[(j,i)]
                    binary_feats[1:1+6,i,j] = binary_feats[1:1+6,j,i] = self.bond_features(bond)
                else:
                    binary_feats[0,i,j] = binary_feats[0,j,i] = 1.0
                binary_feats[-4,i,j] = binary_feats[-4,j,i] = 1.0 if comp[i] != comp[j] else 0.0
                binary_feats[-3,i,j] = binary_feats[-3,j,i] = 1.0 if comp[i] == comp[j] else 0.0
                binary_feats[-2,i,j] = binary_feats[-2,j,i] = 1.0 if n_comp == 1 else 0.0
                binary_feats[-1,i,j] = binary_feats[-1,j,i] = 1.0 if n_comp > 1 else 0.0
        return binary_feats
    
    @staticmethod
    def to_one_hot(codec, values):
        value_idxs = codec.transform(values)
        return torch.eye(len(codec.classes_), dtype=torch.float)[value_idxs]

    #don't need this. In predicting molecule structure, the lables are bonds. Here we use yield. 
    @staticmethod
    def get_bond_labels(edits, n_atoms, sparse_idx):
        bo_index_map = {0.0: 0,
                           1: 1,
                           2: 2,
                           3: 3,
                           1.5: 4}
        edits = edits.split(";")
        bond_labels = torch.zeros((n_atoms, n_atoms, len(bo_index_map)))
        for edit in edits:
            atom1, atom2, bond_order = edit.split("-")
            bo_index = bo_index_map[float(bond_order)]
            bond_labels[int(atom1)-1, int(atom2)-1, bo_index] = bond_labels[int(atom2)-1, int(atom1)-1, bo_index] = 1
        bond_labels = bond_labels[sparse_idx[:,0],sparse_idx[:,1]]
        return bond_labels

    @staticmethod
    def bond_features(bond):
        bt = bond.GetBondType()
        return torch.Tensor(
            [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()])
    
    
