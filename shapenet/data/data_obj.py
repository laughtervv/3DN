'''
    Dataset for shapenet part segmentaion.
'''

import os
import os.path
import json
import numpy as np
import sys
import tensorflow as tf
import numpy as np
import obj


def pc_normalize(pc):
    """ pc: NxC, return NxC """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc

def write_obj(filepath, verts, tris):
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in verts:
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
        for p in tris:
            f.write("f")
            for i in p:
                f.write(" %d" % (i + 1))
            f.write("\n")

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

class ObjDataset():
    def __init__(self, listfile, maxnverts=10000, maxntris=10000, normalize = True, texture=0):
        self.maxnverts = maxnverts
        self.maxntris = maxntris
        self.inputlistfile = listfile
        self.normalize = normalize
        self.texture = texture

        if isinstance(self.inputlistfile, basestring):
            with open(self.inputlistfile, 'r') as f:
                lines = f.read().splitlines()
                self.datapath = [line.strip() for line in lines]
        else:
            self.datapath = listfile

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 18000

    def __getitem__(self, index):

        if self.texture == 0:   
            if index in self.cache:
                verts, tris = self.cache[index]
            else:
                obj_fn = self.datapath[index]+'.obj'
                try:
                    loadobj = obj.load_obj(obj_fn, maxnverts=self.maxnverts, maxntris=self.maxntris)
                except:
                    loadobj = None

                if loadobj == None:
                    if (index + 1) < len(self.datapath):
                        return self.__getitem__(index+1)
                    else:
                        return self.__getitem__(0)
                        
                verts, tris = loadobj
                if self.normalize:
                    verts = pc_normalize(verts)

                if len(self.cache) < self.cache_size:
                    self.cache[index] = (verts, tris)

            return verts, tris
        else:
            if index in self.cache:
                verts, tris, textures = self.cache[index]
            else:
                obj_fn = self.datapath[index]+'.obj'
                try:
                    loadobj = obj.load_obj(obj_fn, texture_size=self.texture, load_texture=True, maxnverts=self.maxnverts, maxntris=self.maxntris)
                except:
                    loadobj = None
                if loadobj == None:
                    if (index + 1) < len(self.datapath):
                        return self.__getitem__(index+1)
                    else:
                        return self.__getitem__(0)

                verts, tris, textures = loadobj
                if self.normalize:
                    verts = pc_normalize(verts)

                if len(self.cache) < self.cache_size:
                    self.cache[index] = (verts, tris, textures)

            return verts, tris, textures


    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':

    d = ObjDataset('/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/part_mesh/Models/filelists/chair.lst')

