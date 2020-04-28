from __future__ import print_function

from .molecule import Mol
import rdkit.Chem.AllChem as AllChem

class Rxn(object):
    """
    """
    def __init__(self, rxn_smiles):
        reactants_smile, reagents_smile, products_smile = rxn_smiles.split('>')
        self.reactants = [Mol(smile) for smile in reactants_smile.split('.')]
        self.reagents = [Mol(smile) for smile in reagents_smile.split('.')]
        self.products = [Mol(smile) for smile in products_smile.split('.')]

    @property
    def reactants_smile(self):
        return '.'.join([mol.smile for mol in self.reactants])

    @property
    def reagents_smile(self):
        return '.'.join([mol.smile for mol in self.reagents])

    @property
    def products_smile(self):
        return '.'.join([mol.smile for mol in self.products])

    @property
    def smile(self):
        return '>'.join([self.reactants_smile, self.reagents_smile, self.products_smile])

    @smile.setter
    def smile(self, smile):
        reactants_smile, reagents_smile, products_smile = smile.split('>')
        self.reactants = [Mol(smile) for smile in reactants_smile.split('.')]
        self.reagents = [Mol(smile) for smile in reagents_smile.split('.')]
        self.products = [Mol(smile) for smile in products_smile.split('.')]

    @property
    def mols(self):
        return self.reactants + self.reagents + self.products

    def canonicalize(self):
        for mol in self.reactants:
            mol.canonicalize()

        for mol in self.reagents:
            mol.canonicalize()

        for mol in self.products:
            mol.canonicalize()

    def remove_rxn_mapping(self):
        rxn = AllChem.ReactionFromSmarts(self.smile, useSmiles=True)
        AllChem.RemoveMappingNumbersFromReactions(rxn)
        self.smile = AllChem.ReactionToSmiles(rxn)
