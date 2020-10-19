from __future__ import print_function

from .molecule import Mol
import rdkit.Chem.AllChem as AllChem

class Rxn(object):
    """
    """
    def __init__(self,product,reactants,r_yield):
        if isinstance(r_yield , float):
            self.yields = float(r_yield)/100
        else: 
            self.yields = -1
        self.reactants = [Mol(r) for r in reactants]
        self.products = [Mol(product)]
        #print('.'.join([mol.smile for mol in self.products]))        
    @property
    def reactants_smile(self):
        return '.'.join([mol.smile for mol in self.reactants])

    @property
    def products_smile(self):
        return '.'.join([mol.smile for mol in self.products])
    
    def get_yield(self):
        return self.yields
    
    @property
    def smile(self):
        return '>'.join([self.reactants_smile, self.products_smile])
    
    @smile.setter
    def smile(self, smile):
        reactants_smile, reagents_smile, products_smile = smile.split('>')
        self.reactants = [Mol(smile) for smile in reactants_smile.split('.')]
        self.reagents = [Mol(smile) for smile in reagents_smile.split('.')]
        self.products = [Mol(smile) for smile in products_smile.split('.')]

    @property
    def mols(self):
        return self.reactants + self.products

    def canonicalize(self):
        for mol in self.reactants:
            mol.canonicalize()

        for mol in self.products:
            mol.canonicalize()

    def remove_rxn_mapping(self):
        rxn = AllChem.ReactionFromSmarts(self.smile, useSmiles=True)
        AllChem.RemoveMappingNumbersFromReactions(rxn)
        self.smile = AllChem.ReactionToSmiles(rxn)

