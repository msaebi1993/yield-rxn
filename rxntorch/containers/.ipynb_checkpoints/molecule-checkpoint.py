from __future__ import print_function

import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors
import torch

class Mol(object):
    """
    """
    def __init__(self, mol_json):
        self.smile = mol_json.get('smiles','')
        self.name= mol_json.get('name','')
        self.category=mol_json.get('category','')
        self.molecular_weight=mol_json.get('molecular_weight',0.0)
        self.volume=mol_json.get('volume',0.0)
        self.surface_area=mol_json.get('surface_area',0.0)
        self.ovality=mol_json.get('ovality',0.0)
        self.hardness=mol_json.get('hardness',0.0)
        self.dipole_moment=mol_json.get('dipole_moment',0.0)
        self.vib_modes=mol_json.get('vib_modes',0.0)
        
        self.electronegativity=mol_json.get('electronegativity',0.0)
        self.HOMO_energy=mol_json.get('HOMO_energy',0.0)
        self.LUMO_energy=mol_json.get('LUMO_energy',0.0)
        self.atoms=mol_json.get('atoms',{})
        self.atoms_nums=len(self.atoms)
        self.attributes=torch.tensor([self.molecular_weight,self.volume,
                                 self.surface_area,self.ovality,
                                 self.hardness,self.dipole_moment,
                                 self.electronegativity,self.HOMO_energy,self.LUMO_energy])
    

    
    def get_attributes(self):
        #print(attributes)
        return self.attributes
    
    def __eq__(self, other):
        return self.smile == other.smile

    def __hash__(self):
        return hash(self.smile)

    @property
    def canonical(self):
        return self.smile == Chem.MolToSmiles(Chem.MolFromSmiles(self.smile))
    #@property
    #def molecular_weight(self):
        #return Descriptors.MolWt(Chem.MolFromSmiles(self.smile))

    def canonicalize(self):
        self.smile = Chem.MolToSmiles(Chem.MolFromSmiles(self.smile))
