import pyrfume
import torch
from rdkit import Chem
import dgllife
from torch.utils.data import Dataset
from dgl.data import DGLDataset

import numpy as np
import torch.nn.functional as F

class LeffingwellDataset(DGLDataset):

    ATOM_TYPES = ['H', 'C', 'S', 'N', 'O']
    BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    N_RANDOM_FEATURES = 5
    NODE_FEAT_LENGTH = 11
    EDGE_FEAT_LENGTH = 4
    N_LABELS = 113


    def __init__(self):
        super().__init__(name='LeffingWell')

    def process(self):
        # Load the data
        molecules = pyrfume.load_data('leffingwell/molecules.csv', remote=True)
        behavior = pyrfume.load_data('leffingwell/behavior.csv', remote=True)

        attribute_indices = behavior.columns[1:]
        attributes = behavior[attribute_indices].to_numpy()
        self.cids = behavior.index.to_numpy()
        all_smiles = behavior[behavior.columns[0]].to_list()

        self.graphs = []
        self.labels = []
        for i in range(len(self.cids)):
            smiles = all_smiles[i]
            labels = attributes[i]
            graph = dgllife.utils.smiles_to_bigraph(smiles, node_featurizer=self.featurize_atoms,
                            edge_featurizer=self.featurize_bonds, explicit_hydrogens=True)
            self.graphs.append(graph)
            self.labels.append(labels)

        self.label_weights = self.calculate_label_weights()

    def calculate_label_weights(self):
        all_graph_labels = self.labels
        all_nonzeros = [x.nonzero()[0] for x in all_graph_labels]
        all_label_numbers = np.concatenate(all_nonzeros)
        all_number_of_labels = [len(x) for x in all_nonzeros]
        label_counts, _ = np.histogram(all_label_numbers, bins=113, range=(0,113))
        label_weights = 1/label_counts
        return label_weights

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)


    def featurize_atoms(self, mol):
        feats = []
        atoms = mol.GetAtoms()
        for atom in atoms:
            feats.append(self.ATOM_TYPES.index(atom.GetSymbol()))
        atom_types =  torch.tensor(feats).long()
        atom_types_onehot = F.one_hot(atom_types, len(self.ATOM_TYPES)).float()
        is_in_ring = torch.tensor([int(atom.IsInRing()) for atom in atoms]).float()
        random_features = torch.randn(len(atoms), self.N_RANDOM_FEATURES)

        return {'h': torch.cat([atom_types_onehot, is_in_ring.unsqueeze(1), random_features], dim=1)}

    def featurize_bonds(self, mol):
        feats = []
        for bond in mol.GetBonds():
            btype = self.BOND_TYPES.index(bond.GetBondType())
            # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
            feats.extend([btype, btype])
        return {'e': F.one_hot(torch.tensor(feats).long(), len(self.BOND_TYPES))}


    def count_number_of_atoms(self, all_smiles):
        # Count the number of each atom
        all_atoms = {}
        for smile in all_smiles:
            mol = Chem.MolFromSmiles(smile)
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol not in all_atoms:
                    all_atoms[symbol] = 1
                else:
                    all_atoms[symbol] += 1
        print(all_atoms)    