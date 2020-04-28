from __future__ import print_function

import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors


class Mol(object):
    """
    """
    def __init__(self, smile):
        self.smile = smile

    def __eq__(self, other):
        return self.smile == other.smile

    def __hash__(self):
        return hash(self.smile)

    @property
    def canonical(self):
        return self.smile == Chem.MolToSmiles(Chem.MolFromSmiles(self.smile))

    @property
    def molecular_weight(self):
        return Descriptors.MolWt(Chem.MolFromSmiles(self.smile))

    def canonicalize(self):
        self.smile = Chem.MolToSmiles(Chem.MolFromSmiles(self.smile))
