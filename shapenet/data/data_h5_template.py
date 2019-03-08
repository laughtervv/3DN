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

class H5Dataset():
    def __init__(self, listfile, maxnverts=10000, maxntris=10000, normalize = False, num_points=2048, batch_size=4):
        self.maxnverts = maxnverts
        self.maxntris = maxntris
        self.inputlistfile = listfile
        self.normalize = normalize
        self.batch_size = batch_size

        with open(self.inputlistfile, 'r') as f:
            lines = f.read().splitlines()
            self.datapath_src = [line.strip().split(' ')[1] for line in lines]
            self.datapath_tgt = [line.strip().split(' ')[0] for line in lines]

        self.data_num = len(lines)
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 60000
        self.order = list(range(self.data_num))


    def get_batch(self, index):

        src_batch_verts = np.zeros((self.batch_size, self.maxnverts, 3))
        src_batch_nverts = np.zeros((self.batch_size, 1)).astype(np.int32)
        src_batch_tris = np.zeros((self.batch_size, self.maxntris, 3)).astype(np.int32)
        src_batch_ntris = np.zeros((self.batch_size, 1)).astype(np.int32)

        tgt_batch_verts = np.zeros((self.batch_size, self.maxnverts, 3))
        tgt_batch_nverts = np.zeros((self.batch_size, 1)).astype(np.int32)
        tgt_batch_tris = np.zeros((self.batch_size, self.maxntris, 3)).astype(np.int32)
        tgt_batch_ntris = np.zeros((self.batch_size, 1)).astype(np.int32)

        cnt = 0
        # for i in range(index, index + self.batch_size):
        i = index
        while cnt < self.batch_size:
            if self.datapath_src[self.order[i]] in self.cache:
                src_single_mesh = self.cache[self.datapath_src[self.order[i]]]
            else:
                src_single_mesh = self.get_one(self.datapath_src[self.order[i]], subsamplemesh=True)
                
            if self.datapath_tgt[self.order[i]] in self.cache:
                tgt_single_mesh = self.cache[self.datapath_tgt[self.order[i]]]
            else:
                tgt_single_mesh = self.get_one(self.datapath_tgt[self.order[i]])

            i += 1
            if i >= self.data_num:
                i = 0

            if tgt_single_mesh == None or src_single_mesh == None:
                continue

            src_verts, src_tris = src_single_mesh
            tgt_verts, tgt_tris = tgt_single_mesh
                
            src_batch_verts[cnt,:len(src_verts),:] = src_verts
            src_batch_tris[cnt,:len(src_tris),:] = src_tris
            src_batch_nverts[cnt, 0] = len(src_verts)
            src_batch_ntris[cnt, 0] = len(src_tris)
            tgt_batch_verts[cnt,:len(tgt_verts),:] = tgt_verts
            tgt_batch_tris[cnt,:len(tgt_tris),:] = tgt_tris
            tgt_batch_nverts[cnt, 0] = len(tgt_verts)
            tgt_batch_ntris[cnt, 0] = len(tgt_tris)

            cnt += 1

        src_batch_data = {}
        src_batch_data['verts'] = src_batch_verts
        src_batch_data['nverts'] = src_batch_nverts
        src_batch_data['tris'] = src_batch_tris
        src_batch_data['ntris'] = src_batch_ntris
        tgt_batch_data = {}
        tgt_batch_data['verts'] = tgt_batch_verts
        tgt_batch_data['nverts'] = tgt_batch_nverts
        tgt_batch_data['tris'] = tgt_batch_tris
        tgt_batch_data['ntris'] = tgt_batch_ntris

        return src_batch_data, tgt_batch_data

    def pc_normalize(self, pc, centroid=None):

        """ pc: NxC, return NxC """
        l = pc.shape[0]

        if centroid is None:
            centroid = np.mean(pc, axis=0)

        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))

        pc = pc / m

        return pc, centroid

    def get_one(self, path, subsamplemesh=False):
        if path in self.cache:
            return self.cache[path]
        else:
            try:
                h5_f = h5py.File(path)
                if (self.maxnverts != -1 and h5_f['verts'].shape[0] > self.maxnverts) or (self.maxntris != -1 and h5_f['tris'].shape[0] > self.maxntris):
                    raise Exception()

                verts, tris = h5_f['verts'][:], h5_f['tris'][:]
            except:
                h5_f.close()
                self.cache[path] = None
                return None

            if self.normalize:
                centroid = None
                verts, _ = self.pc_normalize(verts, centroid)

            if subsamplemesh:
                mesh = pymesh.form_mesh(verts, tris)
                mesh, _ = pymesh.split_long_edges(mesh, 0.05)
                verts, tris = mesh.vertices, mesh.faces
                if (self.maxnverts != -1 and verts.shape[0] > self.maxnverts) or (self.maxntris != -1 and tris.shape[0] > self.maxntris):
                    return None

            if len(self.cache) < self.cache_size:
                self.cache[path] = (verts, tris)
            h5_f.close()

            return verts, tris


    def __len__(self):
        return self.data_num

if __name__ == '__main__':
    MAX_NVERTS = MAX_NTRIS = 5000
    d = H5Dataset('/media/hdd2/data/ShapeNet/filelists/ShapeNetCore.v2.h5/03001627_obj_5000.lst', maxnverts=MAX_NVERTS, maxntris=MAX_NTRIS)

    batch_size = 20
    tic = time.time()
    for i in range(10):
        i = i * batch_size
        batch = d.get_batch(i)
    print (time.time() - tic)



