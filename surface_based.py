import open3d as o3d
import numpy as np
import trimesh
import torch
from torch_geometric.data import Data
import os
import argparse

def reduce_mesh(input_mesh_path, output_mesh_path, reduction_factor=0.02):
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    
    if not mesh.has_triangles():
        print("Error: the 3d object doesn't have any faces !")
        return

    mesh_simplified = mesh.simplify_quadric_decimation(int(len(mesh.triangles) * reduction_factor))
    o3d.io.write_triangle_mesh(output_mesh_path, mesh_simplified)

def get_surface_models(simplified_mesh, file_name, n_points=1024, graphs_output_dir="data/surface_graphs", points_output_dir="data/points"):

    vertices_coor = simplified_mesh.vertices
    vertices_type = simplified_mesh.visual.vertex_colors
    vertices = np.concatenate((vertices_coor, vertices_type[:,:3]), axis=1)
    faces = simplified_mesh.faces
    faces_1 = faces[:,[0,1]]
    faces_2 = faces[:,[0,2]]
    faces_3 = faces[:,[1,2]]

    edges = np.concatenate((faces_1, faces_2, faces_3), axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
        
    vertices = torch.tensor(vertices, dtype=torch.float32)
    edges = torch.tensor(edges, dtype=torch.int64).T
        
    protein_graph_surface = Data(
        x = vertices,
        edge_index = edges,
    )
    torch.save(protein_graph_surface, os.path.join(graphs_output_dir, file_name[:-3]+"pt"))

    points_3d = simplified_mesh.sample(n_points)
    points, face_index = simplified_mesh.sample(n_points, return_index=True)
    colors = simplified_mesh.visual.face_colors[face_index]
    points_3d = np.hstack((points, colors))
    points_3d = points_3d[:, :6]
    points_3d[:, 3:6] = points_3d[:, 3:6] / 255
    torch.save(torch.tensor(points_3d.T), os.path.join(points_output_dir, file_name[:-3]+"pt"))

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=False, default="example.ply", help="PLY Input file name")
    parser.add_argument("-f", "--reduction_factor", type=float, required=False, default=0.02, help="Reduction Factor (Default is 0.02)")
    parser.add_argument("-p", "--n_points", type=int, required=False, default=1024, help="number of points to generate in Point CLouds (Default is 1024)")

    args = parser.parse_args()
    file_name = args.input_file
    reduction_factor = args.reduction_factor
    n_points = args.n_points

    input_mesh_path = f"data/ply/{file_name}"
    output_mesh_path = f"data/reduced/{file_name}"
    reduce_mesh(input_mesh_path, output_mesh_path, reduction_factor=reduction_factor)
    simplified_mesh = trimesh.load(output_mesh_path)
    get_surface_models(simplified_mesh, file_name, n_points=n_points)
    print("✅ Done with success!")