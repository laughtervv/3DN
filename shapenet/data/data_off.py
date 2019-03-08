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

def write_off(fn, verts, faces):
    file = open(fn, 'w')
    file.write('OFF\n')
    file.write('%d %d %d\n' % (len(verts), len(faces), 0))
    for vert in verts:
        file.write('%f %f %f\n' % (vert[0], vert[1], vert[2]))
        # verts.append([float(s) for s in readline().strip().split(' ')])
    for face in faces:
        file.write('3 %d %d %d\n' % (face[0], face[1], face[2]))
        # faces.append([int(s) for s in readline().strip().split(' ')][1:])
    file.close()
    return

def read_off(fn):
    file = open(fn, 'r')
    if 'OFF' != file.readline().strip():
        print ('Not a valid OFF header')
        return
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = []
    for i_vert in range(n_verts):
        verts.append([float(s) for s in file.readline().strip().split(' ')])
    faces = []
    for i_face in range(n_faces):
        faces.append([int(s) for s in file.readline().strip().split(' ')][1:])
    file.close()
    return np.asarray(verts,dtype=np.float32), np.asarray(faces, dtype=np.int32)

def read_lab(fn, ntris, label_dict):

    trilabel = -1 * np.ones(ntris).astype(np.int8)

    with open(fn, 'r') as f:
        lines = f.read().splitlines()

    for i in range(len(lines)):
        s = lines[i].strip()
        if s == '':
            continue
        if s in label_dict.keys():
            i += 1
            tris = [int(st)-1 for st in lines[i].strip().split(' ')]
            trilabel[tris] = label_dict[s]

    return trilabel

def Get_OFF_one_batch(datapath, normalize = True, batch_size=1, maxnverts=None, maxntris=None):
    verts, tris = read_off(datapath)
    if normalize:
        verts = pc_normalize(verts)

    bsize = 1
    batch_nverts = np.zeros((bsize, 1)).astype(np.int32)
    batch_ntris = np.zeros((bsize, 1)).astype(np.int32)

    # for i in range(bsize):
    v1, t1 = verts, tris
    # tl1 = np.ones( t1.shape[0]).astype(np.int32)
    # vl1 = np.ones( v1.shape[0]).astype(np.int32)

    batch_nverts[0,0] = len(v1)
    batch_ntris[0,0] = len(t1)

    batch_data = {}
    # batch_tris = np.zeros((bsize, MAX_NTRIS, 3)).astype(np.int32)
    if maxnverts == None:
        batch_data['verts'] = np.expand_dims(v1, 0)
        batch_data['vertsmask'] = np.ones((1, len(v1), 1)).astype(np.float32)
    else:
        batch_data['verts'] = np.zeros((1, maxnverts, 3))
        batch_data['verts'][:,:len(v1),:] = v1
        batch_data['vertsmask'] = np.zeros((1, maxnverts, 1)).astype(np.float32)
        batch_data['vertsmask'][:, :len(v1), 0] = 1.
    batch_data['nverts'] = batch_nverts

    if maxnverts == None:
        batch_data['tris'] = np.expand_dims(t1, 0)
        batch_data['trismask'] = np.ones((1, len(t1), 1)).astype(np.float32)
    else:
        batch_data['tris'] = np.zeros((1, maxntris, 3))
        batch_data['tris'][:,:len(t1),:] = t1
        batch_data['trismask'] = np.zeros((1, maxntris, 1)).astype(np.float32)
        batch_data['trismask'][:, :len(t1), 0] = 1.
    batch_data['ntris'] = batch_ntris

    if batch_size > 1:
        batch_data['verts'] = np.tile(batch_data['verts'], (batch_size, 1, 1))
        batch_data['nverts'] = np.tile(batch_data['nverts'], (batch_size, 1))
        batch_data['tris'] = np.tile(batch_data['tris'], (batch_size, 1, 1))
        batch_data['ntris'] = np.tile(batch_data['ntris'], (batch_size, 1))
        batch_data['vertsmask'] = np.tile(batch_data['vertsmask'], (batch_size, 1, 1))
        batch_data['trismask'] = np.tile(batch_data['trismask'], (batch_size, 1, 1))

    return batch_data

class OFFDataset():
    def __init__(self, listfile, catlist = None, maxnverts=10000, maxntris=10000, normalize = True):
        self.maxnverts = maxnverts
        self.maxntris = maxntris
        self.inputlistfile = listfile
        self.catlistfile = catlist
        self.normalize = normalize

        with open(self.inputlistfile, 'r') as f:
            lines = f.read().splitlines()
            self.datapath = [line.strip() for line in lines]
        if catlist is not None:
            with open(self.catlistfile, 'r') as f:
                lines = f.read().splitlines()
                cat = [line.strip() for line in lines]
                self.cat_dict = dict(zip(cat, range(len(cat))))
        else:
            self.cat_dict = None

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 18000

    def __getitem__(self, index):
        if self.cat_dict is not None:
            if index in self.cache:
                verts, tris, vertslabel, trilabel = self.cache[index]
            else:
                off = self.datapath[index]+'.off'
                verts, tris = read_off(off)
                if len(verts) > self.maxnverts or len(tris) > self.maxntris:
                    if (index + 1) < len(self.datapath):
                        # print 'exceed max nverts', index+1
                        return self.__getitem__(index+1)
                    else:
                        return self.__getitem__(0)

                lab = self.datapath[index]+'.lab'
                trilabel = read_lab(lab, len(tris), self.cat_dict).astype(np.int32)

                vertslabel = -1 * np.ones(len(verts)).astype(np.int8)
                vertsind = tris[:, 0]
                vertslabel[vertsind] = trilabel
                vertsind = tris[:, 1]
                vertslabel[vertsind] = trilabel
                vertsind = tris[:, 2]
                vertslabel[vertsind] = trilabel

                if self.normalize:
                    verts = pc_normalize(verts)

                if len(self.cache) < self.cache_size:
                    self.cache[index] = (verts, tris, vertslabel, trilabel)

            return verts, tris, vertslabel,  trilabel#, index
        else:
            if index in self.cache:
                verts, tris = self.cache[index]
            else:
                off = self.datapath[index]+'.off'
                verts, tris = read_off(off)
                if len(verts) > self.maxnverts or len(tris) > self.maxntris:
                    if (index + 1) < len(self.datapath):
                        return self.__getitem__(index+1)
                    else:
                        return self.__getitem__(0)

                if self.normalize:
                    verts = pc_normalize(verts)

                if len(self.cache) < self.cache_size:
                    self.cache[index] = (verts, tris)

            return verts, tris


    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    # chairpath = '/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/part_mesh/Models/Chair'
    # chairfiles = os.listdir(chairpath)
    # nverts = []
    # ntris = []
    # f=open('/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/part_mesh/Models/filelists/chair.lst', 'w')
    # for fn in chairfiles:
    #     if fn.endswith('off'):
    #         f.write('%s\n'% os.path.join(chairpath,fn.strip('.off')))
    #         # verts, tris = read_off(os.path.join(chairpath, fn))
    #         # nverts += [verts]
    #         # ntris += [tris]


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
    from utils import show3d_balls
    labfile = '/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/part_mesh/Models/Chair/1e1151a459002e85508f812891696df0.lab'
    offfile = '/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/part_mesh/Models/Chair/1e1151a459002e85508f812891696df0.off'
    verts, tris = read_off(offfile)
    label_dict={'Chair_labelA':0,'Chair_labelB':1,'Chair_labelC':2,'Chair_labelD':3}
    color_dict=[[1,0,0],[0,1,0],[0,0,1],[0,0.5,0.5]]
    lab = read_lab(labfile, len(tris),label_dict)
    print (lab)

    with tf.Session('') as sess:
        dst_verts = tf.expand_dims(tf.constant(verts), 0)
        dst_tris = tf.expand_dims(tf.constant(tris), 0)
        dst_feats = tf.expand_dims(tf.constant(verts), 0)
        dst_nverts = tf.constant([[len(verts)]], dtype=tf.int32)
        dst_ntris = tf.constant([[len(tris)]], dtype=tf.int32)
        r1 = tf.random_uniform([1, 40000],dtype=tf.float32)
        r2 = tf.random_uniform([1, 40000],dtype=tf.float32)
        r = tf.random_uniform([1, 40000],dtype=tf.float32)
        dst_points, outfeats, correspondingfaces = tf_meshsampling.mesh_sampling(dst_verts, dst_nverts, dst_tris, dst_ntris,
                                                                                 dst_feats, r, r1, r2)
        dst_points_val, dst_verts_val = sess.run([dst_points, dst_verts])

        B = np.zeros([1, 40000, 1], dtype=np.int32)
        for iB in range(1):
            B[iB, :, :] = iB
        B = tf.constant(B)

        correspondingfaces = tf.expand_dims(correspondingfaces, 2)
        ref_indices = tf.concat(axis=2, values=[B, correspondingfaces])
        ref_tri_part = tf.expand_dims(tf.constant(lab), 0)
        ref_part_mask = tf.gather_nd(ref_tri_part, ref_indices, name=None)

        d = PartMeshDataset('/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/part_mesh/Models/filelists/chair.lst',
                            '/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/part_mesh/Models/filelists/chair_cat.lst')


        points_val,ref_part_mask_val = sess.run([dst_points,ref_part_mask])

        points_val = np.squeeze(points_val)
        ref_part_mask_val = np.squeeze(ref_part_mask_val)
        c_gt = np.zeros([len(ref_part_mask_val),3])
        for i in range(len(ref_part_mask_val)):
            c_gt[i] = color_dict[ref_part_mask_val[i]]
        show3d_balls.showpoints(points_val, c_gt=c_gt, ballradius=2)

    # d = PartDataset(root=os.path.join(BASE_DIR, 'data/shapenetcore_partanno_segmentation_benchmark_v0'),
    #                 class_choice=['Chair'], split='trainval')
    # print(len(d))
    # import time
    #
    # tic = time.time()
    # i = 100
    # ps, seg = d[i]
    # print(np.max(seg), np.min(seg))
    # print(time.time() - tic)
    # print(ps.shape, type(ps), seg.shape, type(seg))
    # sys.path.append('utils')
    # import show3d_balls
    #
    #
    # d = PartMeshDataset(root=os.path.join(BASE_DIR, 'data/shapenetcore_partanno_segmentation_benchmark_v0'),
    #                 classification=True)
    # print(len(d))
    # ps, cls = d[0]
    # print(ps.shape, type(ps), cls.shape, type(cls))

