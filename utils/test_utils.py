import numpy as np 
import os
import pymesh

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def iou_pymesh(mesh1, mesh2, dim=110):
    # mesh1 = (vertices1, triangles1)
    # mesh2 = (vertices2, triangles2)

    mesh1 = pymesh.form_mesh(mesh1[0], mesh1[1])
    grid1 = pymesh.VoxelGrid(2./dim)
    grid1.insert_mesh(mesh1)
    grid1.create_grid()

    ind1 = ((grid1.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
    v1 = np.zeros([dim, dim, dim])
    v1[ind1[:,0], ind1[:,1], ind1[:,2]] = 1


    mesh2 = pymesh.form_mesh(mesh2[0], mesh2[1])
    grid2 = pymesh.VoxelGrid(2./dim)
    grid2.insert_mesh(mesh2)
    grid2.create_grid()

    ind2 = ((grid2.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
    v2 = np.zeros([dim, dim, dim])
    v2[ind2[:,0], ind2[:,1], ind2[:,2]] = 1

    intersection = np.sum(np.logical_and(v1, v2))
    union = np.sum(np.logical_or(v1, v2))
    return float(intersection) / union

def iou_binvox(off_path1, off_path2):

    solidbinvoxpath1 = off_path1[:-4] + '.binvox'
    if not os.path.exists(solidbinvoxpath1):
        os.system("%s/binvox -cb -dc -aw -bb -1. -1. -1. 1. 1. 1. -d 128 %s -pb 2>&1 >/dev/null" % (BASE_DIR, off_path1))
    with open(solidbinvoxpath1, 'rb') as f:
        v1 = binvox_rw.read_as_3d_array(f)#.data
        pc1 = process_binvox(v1)

    solidbinvoxpath2 = off_path2[:-4] + '.binvox'
    if not os.path.exists(solidbinvoxpath2):
        os.system("%s/binvox -cb -dc -aw -bb -1. -1. -1. 1. 1. 1. -d 128 %s -pb 2>&1 >/dev/null" % (BASE_DIR, off_path2))  
    with open(solidbinvoxpath2, 'rb') as f:
        v2 = binvox_rw.read_as_3d_array(f)#.data
        pc2 = process_binvox(v2)

    intersection = np.sum(np.logical_and(v1.data, v2.data))
    union = np.sum(np.logical_or(v1.data, v2.data))
    return float(intersection) / union


if __name__ == "__main__":

    off_path1 = '/media/ssd/projects/Deformation/Sources/3DN/shapenet/3D/checkpoint/3DN_allcategories_ft/test_results/0_deformmesh.obj'
    off_path2 = '/media/ssd/projects/Deformation/Sources/3DN/shapenet/3D/checkpoint/3DN_allcategories_ft/test_results/0_refmesh.obj'

    mesh1 = pymesh.load_mesh('/media/ssd/projects/Deformation/Sources/3DN/shapenet/3D/checkpoint/3DN_allcategories_ft/test_results/0_deformmesh.obj');
    mesh2 = pymesh.load_mesh('/media/ssd/projects/Deformation/Sources/3DN/shapenet/3D/checkpoint/3DN_allcategories_ft/test_results/0_refmesh.obj');
    print(iou_pymesh((mesh1.vertices, mesh1.faces), (mesh2.vertices, mesh2.faces)))
    # grid = pymesh.VoxelGrid(0.01);
    # grid.insert_mesh(mesh);
    # grid.create_grid();
    # out_mesh = grid.mesh;
    # print(out_mesh.vertices)
    # pymesh.save_mesh('tmp.obj', out_mesh);
    # print(iou_off(off_path1, off_path2))
