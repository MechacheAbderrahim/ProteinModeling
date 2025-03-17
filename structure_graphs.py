import numpy as np
import os
import pandas as pd
import torch
from torch_geometric.data import Data
from Bio.PDB import *
from rdkit import Chem
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from rdkit import Chem
import math
import argparse

def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

aa_to_num = {
    "ALA": 1, "ARG": 2, "ASN": 3, "ASP": 4, "CYS": 5,
    "GLN": 6, "GLU": 7, "GLY": 8, "HIS": 9, "ILE": 10,
    "LEU": 11, "LYS": 12, "MET": 13, "PHE": 14, "PRO": 15,
    "SER": 16, "THR": 17, "TRP": 18, "TYR": 19, "VAL": 20
}

def get_nodes(mol):
    atom_infos = []
    conformer = mol.GetConformer()
    
    for atom in mol.GetAtoms():
        pos = conformer.GetAtomPosition(atom.GetIdx())
        res_info = atom.GetPDBResidueInfo()
        
        if res_info:
            atom_info = {
                'Index': atom.GetIdx(),
                'Nom de l\'Atome': 1 if res_info.GetName().strip() == "CA" else 0,  # Utilisation du nom PDB complet
                #'Symbole': atom.GetSymbol(),
                'Numéro Atomique': atom.GetAtomicNum(),
                'x': float(pos.x),
                'y': float(pos.y),
                'z': float(pos.z),
                'Nom du Résidu': aa_to_num[res_info.GetResidueName().strip()],
                'Numéro de Résidu': res_info.GetResidueNumber()
            }
            atom_infos.append(list(atom_info.values()))

    return atom_infos

def get_bonds(mol):  
    bonds_infos = [[],[]]
    for bond in mol.GetBonds():
        atome1 = bond.GetBeginAtom()
        atome2 = bond.GetEndAtom()
        bonds_infos[0].append(atome1.GetIdx())
        bonds_infos[1].append(atome2.GetIdx())
    return bonds_infos

def clean_pdb(pdb_path):
    with open(pdb_path, 'r') as file:
        lines = file.readlines()
    
    with open(pdb_path, 'w') as file:
        for line in lines:
            line = line.replace("SEP", "SER")  # Phosphosérine → Sérine
            line = line.replace("HIP", "HIS")  # Histidine protonée → Histidine
            file.write(line)

def get_final_graph(file_name, nodes, bonds, points):
    graph = nodes
    cords_1 = graph[:,3:6]
    cords_2 = points.T[:,:3].numpy()

    indexs = []
    for p in range(len(cords_2)): # 1024
        min_d = euclidean_distance(cords_1[0], cords_2[p])
        min_i = 0
        for g in range(1, len(cords_1)):
            d = euclidean_distance(cords_1[g], cords_2[p])
            if min_d > d:
                min_d = d
                min_i = g
        indexs.append(min_i)

    zeros = np.zeros((graph.shape[0], 1))
    new_graph = np.concatenate((graph, zeros), axis=1)

    for idx in indexs:
        new_graph[idx][-1] += 1
    
    x = torch.tensor(new_graph, dtype=torch.float32)
    edges = torch.tensor(bonds, dtype=torch.long)


    graph_data = Data(
        x = x,
        edge_index=edges,
        name = file_name
    )
    torch.save(graph_data, f"data/structure_graphs/{file_name}.pt")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=False, default="example.pdb", help="PDB Input file name")
    args = parser.parse_args()
    file_name = args.input_file[:-4] # delete extension '.pdb'

    pdb_path = f'data/pdb/{file_name}.pdb'
    clean_pdb(pdb_path)
    points = torch.load(f'data/points/{file_name}.pt')

    try:
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        if mol is None:
            raise ValueError("The molecule could not be loaded. Check the PDB file.")

        nodes = np.array(get_nodes(mol))
        bonds = np.array(get_bonds(mol))
        get_final_graph(file_name, nodes=nodes, bonds=bonds, points=points)
    except:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            nodes = np.array(get_nodes(mol))
            bonds = np.array(get_bonds(mol))
            get_final_graph(file_name, nodes=nodes, bonds=bonds, points=points)
                    
        except Exception as e:
            print("❌ Error: Output file was not generated.")
