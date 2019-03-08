'''
    Dataset for shapenet part segmentaion.
'''

import os
import os.path
import json
import numpy as np
import sys
import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import colors
# from matplotlib.ticker import PercentFormatter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR))
sys.path.append(os.path.join(ROOT_DIR,'..'))
try:
    import models.tf_ops.approxmatch.tf_approxmatch as tf_approxmatch
    import models.tf_ops.nn_distance.tf_nndistance as tf_nndistance
    import models.tf_ops.mesh_sampling.tf_meshsampling as tf_meshsampling
    import models.tf_ops.mesh_laplacian.tf_meshlaplacian as tf_meshlaplacian
except:
    import models.tf_ops_server.approxmatch.tf_approxmatch as tf_approxmatch
    import models.tf_ops_server.nn_distance.tf_nndistance as tf_nndistance
    import models.tf_ops_server.mesh_sampling.tf_meshsampling as tf_meshsampling
    import models.tf_ops_server.mesh_laplacian.tf_meshlaplacian as tf_meshlaplacian


def pc_normalize(pc):
    """ pc: NxC, return NxC """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


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



class PartPcDataset():
    def __init__(self, listfile, npoints, normalize = True):
        self.inputlistfile = listfile
        self.normalize = normalize

        with open(self.inputlistfile, 'r') as f:
            lines = f.read().splitlines()
            self.datapath = [line.strip() for line in lines]

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 18000
        self.npoints = npoints

    def __getitem__(self, index):
        if index in self.cache:
            points, label = self.cache[index]
        else:
            fn = self.datapath[index]+'.pts'
            points = np.loadtxt(fn)
            fn_label = self.datapath[index].replace("points", "points_label")+'.seg'
            label = np.loadtxt(fn_label) - 1
            choice = np.random.choice(len(points), self.npoints, replace=True)
            # resample
            try:
                points = points[choice, :]
                label = label[choice]
            except:
                if (index + 1) < len(self.datapath):
                    # print 'exceed max nverts', index+1
                    return self.__getitem__(index+1)
                else:
                    return self.__getitem__(0)
            if self.normalize:
                points = pc_normalize(points)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (points, label)

        return points, label#, index

    def __len__(self):
        return len(self.datapath)
