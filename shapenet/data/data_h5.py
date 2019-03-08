'''
    Dataset for shapenet part segmentaion.
'''

import os
import os.path
import json
import numpy as np
import sys
import time
import tensorflow as tf
import numpy as np
import h5py
import pymesh



def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in np.arange(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

class H5Dataset():
    def __init__(self, listfile, maxnverts=10000, maxntris=10000, normalize = True, texture=0, load_surfacebinvox=False, load_solidbinvox=False, num_points=2048, batch_size=4, test=False, subsamplemesh=False):
        self.maxnverts = maxnverts
        self.maxntris = maxntris
        self.inputlistfile = listfile
        self.normalize = normalize
        self.texture = texture
        self.load_surfacebinvox = load_surfacebinvox
        self.load_solidbinvox = load_solidbinvox
        self.num_points = num_points
        self.batch_size = batch_size
        self.subsamplemesh = subsamplemesh
        self.test = test

        if isinstance(self.inputlistfile, str):
            with open(self.inputlistfile, 'r') as f:
                lines = f.read().splitlines()
                self.datapath = [line.strip() for line in lines]
        else:
            self.datapath = listfile

        self.data_num = len(self.datapath)
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 60000
        self.order = list(range(len(self.datapath)))


    def get_batch(self, index):
        # print index
        if index + self.batch_size > self.data_num:
            index = index + self.batch_size - self.data_num

        batch_verts = np.zeros((self.batch_size, self.maxnverts, 3))
        batch_nverts = np.zeros((self.batch_size, 1)).astype(np.int32)
        batch_tris = np.zeros((self.batch_size, self.maxntris, 3)).astype(np.int32)
        batch_ntris = np.zeros((self.batch_size, 1)).astype(np.int32)
        batch_vertsmask = np.zeros((self.batch_size, self.maxnverts, 1)).astype(np.float32)
        batch_trismask = np.zeros((self.batch_size, self.maxntris, 1)).astype(np.float32)

        if self.texture != 0:
            batch_textures = np.zeros((self.batch_size, self.maxntris, self.texture, self.texture, self.texture)).astype(np.float32)
        else:
            batch_textures = None

        if self.load_solidbinvox != 0:
            batch_solidbinvoxpc = np.zeros((self.batch_size, self.num_points, 3)).astype(np.float32)
        else:
            batch_solidbinvoxpc = None

        if self.load_surfacebinvox != 0:
            batch_surfacebinvoxpc = np.zeros((self.batch_size, self.num_points, 3)).astype(np.float32)
        else:
            batch_surfacebinvoxpc = None


        cnt = 0
        # for i in range(index, index + self.batch_size):
        i = index
        while cnt < self.batch_size:
            if self.order[i] in self.cache:
                single_mesh = self.cache[self.order[i]]
            else:
                single_mesh = self.__getitem__(self.order[i])
                if len(self.cache) < self.cache_size:
                    self.cache[self.order[i]] = single_mesh
            i += 1
            if i >= len(self.datapath):
                i = 0
            if single_mesh == None:
                continue
            v1, t1, text1, surface1, solid1 = single_mesh
                
            batch_verts[cnt,:len(v1),:] = v1
            batch_tris[cnt,:len(t1),:] = t1
            batch_nverts[cnt,0] = len(v1)
            batch_ntris[cnt,0] = len(t1)
            batch_vertsmask[cnt,:len(v1),0] = 1.
            batch_trismask[cnt,:len(t1),0] = 1.

            if self.load_solidbinvox != 0:
                choice = np.random.randint(solid1.shape[0], size=self.num_points)
                batch_solidbinvoxpc[cnt,:,:] = solid1[choice, :]

            if self.load_surfacebinvox != 0:
                choice = np.random.randint(surface1.shape[0], size=self.num_points)
                batch_surfacebinvoxpc[cnt,:,:] = surface1[choice, :]

            cnt += 1


        batch_data = {}
        batch_data['verts'] = batch_verts
        batch_data['nverts'] = batch_nverts
        batch_data['tris'] = batch_tris
        batch_data['ntris'] = batch_ntris
        batch_data['vertsmask'] = batch_vertsmask
        batch_data['trismask'] = batch_trismask
        batch_data['solidbinvoxpc'] = batch_solidbinvoxpc
        batch_data['surfacebinvoxpc'] = batch_surfacebinvoxpc

        return batch_data

    def pc_normalize(self, pc, centroid=None):

        """ pc: NxC, return NxC """
        l = pc.shape[0]

        if centroid is None:
            centroid = np.mean(pc, axis=0)

        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))

        pc = pc / m

        return pc, centroid

    def __getitem__(self, index):
        if index in self.cache:
            verts, tris, textures, surfacebinvoxpc, solidbinvoxpc = self.cache[index]
        else:
            try:
                h5_f = h5py.File(self.datapath[index])
                if (self.maxnverts != -1 and h5_f['verts'].shape[0] > self.maxnverts) or (self.maxntris != -1 and h5_f['tris'].shape[0] > self.maxntris):
                    h5_f.close()
                    raise Exception()
                if self.load_surfacebinvox != 0 and ('surfacebinvoxsparse' not in h5_f.keys()):
                    raise Exception()

                verts, tris = h5_f['verts'][:], h5_f['tris'][:]
            except:
                self.cache[index] = None
                return None


            if self.texture != 0:
                textures = h5_f['textures'][:]
            else:
                textures = None

            if self.load_surfacebinvox != 0:
                surfacebinvoxpc = h5_f['surfacebinvoxsparse'][:].T
            else:
                surfacebinvoxpc = None

            if self.load_solidbinvox != 0:
                solidbinvoxpc = h5_f['binvoxsolid'][:].T
            else:
                solidbinvoxpc = None

            if self.normalize:
                centroid = None

                if self.load_solidbinvox != 0:
                    solidbinvoxpc, centroid = self.pc_normalize(solidbinvoxpc)

                if self.load_surfacebinvox != 0:
                    surfacebinvoxpc, centroid = self.pc_normalize(surfacebinvoxpc)

                verts, _ = self.pc_normalize(verts, centroid)

            if self.test:
                mesh = pymesh.form_mesh(verts, tris)
                mesh, _ = pymesh.split_long_edges(mesh, 0.05)
                verts, tris = mesh.vertices, mesh.faces
                if (self.maxnverts != -1 and verts.shape[0] > self.maxnverts) or (self.maxntris != -1 and tris.shape[0] > self.maxntris):
                    return None

            if self.subsamplemesh:
                mesh = pymesh.form_mesh(verts, tris)
                mesh, _ = pymesh.split_long_edges(mesh, 0.05)
                verts, tris = mesh.vertices, mesh.faces
                if (self.maxnverts != -1 and verts.shape[0] > self.maxnverts) or (self.maxntris != -1 and tris.shape[0] > self.maxntris):
                    return None

            if len(self.cache) < self.cache_size:
                self.cache[index] = (verts, tris, textures, surfacebinvoxpc, solidbinvoxpc)
            h5_f.close()

        return verts, tris, textures, surfacebinvoxpc, solidbinvoxpc


    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    MAX_NVERTS = MAX_NTRIS = 5000
    def get_batch_mesh(dataset, bsize):
        # bsize = 1
        batch_verts = np.zeros((bsize, MAX_NVERTS, 3))
        batch_nverts = np.zeros((bsize, 1)).astype(np.int32)
        batch_tris = np.zeros((bsize, MAX_NTRIS, 3)).astype(np.int32)
        batch_ntris = np.zeros((bsize, 1)).astype(np.int32)

        for i, i_d in enumerate(np.random.randint(len(dataset), size=bsize)):
            # v1, t1, vl1, tl1= dataset[i_d]
            v1, t1 = dataset[i_d]
            batch_verts[i,:len(v1),:] = v1
            batch_tris[i,:len(t1),:] = t1
            batch_nverts[i,0] = len(v1)
            batch_ntris[i,0] = len(t1)

        # batch_verts[bsize-1,:len(rect['verts'][0,...]),:] = rect['verts'][0,...]
        # batch_tris[bsize-1,:len(rect['tris'][0,...]),:] = rect['tris'][0,...]
        # batch_nverts[bsize-1,0] = rect['nverts'][0,...]
        # batch_ntris[bsize-1,0] = rect['ntris'][0,...]

        batch_data = {}
        batch_data['verts'] = batch_verts
        batch_data['nverts'] = batch_nverts
        batch_data['tris'] = batch_tris
        batch_data['ntris'] = batch_ntris

        return batch_data
    d = H5Dataset('/media/hdd2/data/ShapeNet/filelists/ShapeNetCore.v2.h5/03001627_obj_5000.lst', maxnverts=MAX_NVERTS, maxntris=MAX_NTRIS)

    batch_size = 20
    tic = time.time()
    for i in range(10):
        batch = get_batch_mesh(d, batch_size)
    print (time.time() - tic)



