import numpy as np 
import torch
import open3d as o3d
import argparse
import trimesh
import pdb

def _get_o3d_mesh(verts: np.ndarray, faces: np.ndarray):
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces),
    )
    assert(mesh.has_vertices())
    return mesh

def decimate_mesh_o3d(vertices, faces, decimate_rate):
    numF_gt = len(faces)
    numF_target = int(numF_gt * (1-decimate_rate))
    o3d_mesh_in = _get_o3d_mesh(vertices, faces)
    o3d_mesh_smp = o3d_mesh_in.simplify_quadric_decimation(target_number_of_triangles=numF_target)
    verts, faces = np.asarray(o3d_mesh_smp.vertices), np.asarray(o3d_mesh_smp.triangles)
    return verts, faces


def save_mesh_to_ply(vertices, colors, normals, faces, ply_filename):
    assert len(vertices) == len(colors) == len(normals), "All arrays must have the same length."

    # Create a trimesh mesh object
    mesh = trimesh.Trimesh(vertices=vertices,
                           vertex_colors=colors,
                           vertex_normals=normals,
                           faces=faces)

    # Export the mesh to a PLY file
    mesh.export(ply_filename)

parser = argparse.ArgumentParser(description='Process COLMAP sparse reconstruction')
parser.add_argument("--mesh_file", type=str, help="Mesh file to be processed")
parser.add_argument("--out_file", type=str, help="Output file after SOR")
parser.add_argument("--decimate_rate", type=float, help="Decimation rate")
args = parser.parse_args()

use_decimate = True
device = torch.device('cuda')
decimate_rate = args.decimate_rate
mesh_file = args.mesh_file 
out_file = args.out_file
print('Processing: ', mesh_file)
print('Output: ', out_file)
print('Decimate rate: ', decimate_rate)

mesh = trimesh.load(mesh_file, force='mesh', skip_material=True, process=False)
vertices, faces = mesh.vertices, mesh.faces

if use_decimate:
    vertices, faces = decimate_mesh_o3d(vertices, faces, decimate_rate=decimate_rate)
    mesh = trimesh.Trimesh(vertices, faces)

vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
vertex_normals = torch.tensor(mesh.vertex_normals, dtype=torch.float, device=device)
vertex_rgbs = torch.ones_like(vertex_normals) * 255
# vertex_rgbs = torch.tensor(mesh.visual.vertex_colors)[:,:3]

save_mesh_to_ply(vertices.cpu().numpy().astype(np.float32), vertex_rgbs.cpu().detach().numpy().astype(np.int32),
                    vertex_normals.cpu().numpy().astype(np.float32), faces.cpu().numpy().astype(np.int32), out_file)

print('done!!')


