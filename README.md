# Protein Modeling

This repository contains the scripts used to generate the models proposed in our work entitled:  
**"Comparative Analysis of Different Modeling Approaches for Protein Classification"**.

## Scripts

### EDTSurf.py
Generates a 3D surface mesh (PLY) (`.data/ply`) from a PDB file (`.data/pdb`) using EDTSurf (Linux version).  
Default EDTSurf parameters are used. For more information, refer to the official EDTSurf website:  
https://zhanggroup.org/EDTSurf/

Example usage: `python EDTSurf.py -i example.pdb`

### surface_based.py
From a PLY file, this script creates two surface-based representations:  
- A point cloud `./data/points`
- A surface graph  `./data/surface_graphs`

The outputs are saved as PyTorch tensors (.pt).

Example usage: `python surface_based.py -i example.ply -f 0.1 -p 1024`  
`-f` is the reduction factor, `-p` is the number of points to generate for the point cloud.

### structure_graphs.py
Generates an atomic-level structure graph (`./data/structure_graphs`) from a PDB file and its corresponding point cloud (.pt).  
To generate a residue-level graph, you need to select only the **Cα atoms** of each amino acid.

The `example.ply` file must be available in `./data/ply`

Example usage: `python structure_graphs.py -i example.pdb`

## Data
- The Data used in this work is avalaible at http://shrec2019.drugdesign.fr
## Notes
- The EDTSurf binary is located in the `/bin` folder and must be executable on Linux.
- A sample PDB file is available in `./data/pdb` for testing.
- The model used to train the **graph-based representations** is implemented in `.models/gcnn.py`.
- The model used for the **point cloud representations** is based on PointNet:  
  Paper: https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf  
  PyTorch implementation: https://github.com/fxia22/pointnet.pytorch  
  TensorFlow implementation: https://github.com/charlesq34/pointnet

## ⚠️ Important
Before running the scripts, be sure to create the following folders inside the `./data/` directory:  
`ply/`, `points/`, `reduced/`, `structure_graphs/`, and `surface_graphs/`.

You can create them all at once using:  
`mkdir -p ./data/ply ./data/points ./data/reduced ./data/structure_graphs ./data/surface_graphs`
