'''
    Dataset for shapenet part segmentaion.
'''

import os
import json
import numpy as np
import sys
import time
import random
import tensorflow as tf
import numpy as np
import h5py
import cv2

cam_dict = {
    'r_000':[0.000000, 1.000000, 1.184425, 0.383219, -1.292325, 1.000000, 0.000000, 0.000000, 0.000000],
    'r_030':[330.000000, 1.000000, 1.108089, 0.306618, -1.294921, -0.965926, 0.000000, 0.258819, 0.000000 ],
    'r_060':[300.000000, 1.000000, 1.519251, 0.290514, -1.680355, -0.866025, 0.000000, 0.500000, 0.000000 ],
    'r_090':[270.000000, 1.000000, 0.162809, -0.067033, -0.053511, -0.707107, 0.000000, 0.707107, 0.000000 ],
    'r_120':[240.000000, 1.000000, 1.680631, 0.328789, -1.717884, -0.500000, 0.000000, 0.866025, 0.000000 ],
    'r_150':[210.000000, 1.000000, 0.803019, 0.415971, -0.905644, -0.258819, 0.000000, 0.965926, 0.000000 ],
    'r_180':[180.000000, 1.000000, 1.404542, 0.385169, -1.344790, 0.000000, 0.000000, 1.000000, 0.000000 ],
    'r_210':[150.000000, 1.000000, 1.700142, 0.298616, -1.557647, 0.258819, 0.000000, 0.965926, 0.000000 ],
    'r_240':[120.000000, 1.000000, 1.143554, 0.182567, -1.083467, 0.500000, 0.000000, 0.866025, 0.000000 ],
    'r_270':[90.000000, 1.000000, -0.214220, -0.135983, 0.127193, 0.707107, 0.000000, 0.707107, 0.000000 ],
    'r_300':[60.000000, 1.000000, 0.217362, 0.446668, 0.359680, 0.866025, 0.000000, 0.500000, 0.000000 ],
    'r_330':[30.000000, 1.000000, 0.614698, 0.351053, -0.480797, 0.965926, 0.000000, 0.258819, 0.000000 ],
}

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

def memory():
    """
    Get node total memory and memory usage
    """
    with open('/proc/meminfo', 'r') as mem:
        ret = {}
        tmp = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemTotal:':
                ret['total'] = int(sline[1])
            elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                tmp += int(sline[1])
        ret['free'] = tmp
        ret['used'] = int(ret['total']) - int(ret['free'])
    return ret

class RenderedImgDataset():
    def __init__(self, listfile, batch_size=8, img_size=(600,600), maxnverts=10000, maxntris=10000, normalize = True, texture=0, h5folder='/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/ShapeNetCore.v2.h5'):

        self.maxnverts = maxnverts
        self.maxntris = maxntris
        self.inputlistfile = listfile
        self.normalize = normalize
        self.texture = texture
        self.h5folder = h5folder
        self.img_size = img_size
        self.batch_size = batch_size

        if isinstance(self.inputlistfile, str):
            with open(self.inputlistfile, 'r') as f:
                lines = f.read().splitlines()
                self.datapath = [line.strip() for line in lines]
        else:
            self.datapath = listfile

        self.order = list(range(len(self.datapath)))
        # random.shuffle(self.order)

        self.cache_mesh = {}
        self.cache_img = {}

        self.current = 0

    def __getitem__(self, index):

        img = np.asarray(Image.open(self.datapath[index]), dtype=np.float32)
        mask = img[:,:,3]
        img = img[:,:,:3] / 255.
        img[mask==0,:] = 1.

        modelname = self.datapath[index].split('/')[-3]

        if modelname in self.cache:
            mesh = self.cache[modelname]

        names = self.datapath[index].split('/')
        cat = names[-4]

        ##load mesh from h5
        h5path = os.path.join(self.h5folder, cat, modelname+'.h5')
        h5_f = h5py.File(h5path)  

        if self.texture == 0:  
            verts, tris = h5_f['verts'][:], h5_f['tris'][:]
            if self.normalize:
                verts = pc_normalize(verts)
            mesh = (verts, tris)
        else:
            verts, tris, textures = h5_f['verts'][:], h5_f['tris'][:], h5_f['textures'][:]
            if self.normalize:
                verts = pc_normalize(verts)
            mesh = (verts, tris, textures)
        h5_f.close()

        # get camera parameters for nerual renderer
        cam_id = names[-1].strip('.png')
        cam = cam_dict[cam_id]

        mem = memory()
        if mem['free'] < 0.1 * mem['total']:
            self.cache[modelname] = mesh

        return img, mesh, cam

    def get_img(self, index):

        if self.datapath[index] in self.cache_img:
            img, mask, cam = self.cache_img[self.datapath[index]]
        else:
            names = self.datapath[index].split('/')
            cam_id = names[-1].strip('.png')
            if cam_id  in cam_dict:
                cam = cam_dict[cam_id]
            else:
                cam = None

            # tic = time.time()
            img = cv2.imread(self.datapath[index], cv2.IMREAD_UNCHANGED)
            if self.img_size[0] != img.shape[0] or self.img_size[1] != img.shape[1]:
                print img.shape
                img = cv2.resize(img, self.img_size).astype(np.float32)
            # print 'img', time.time() - tic

            if img.shape[2] == 4:
                mask = img[:,:,3]
                img = img[:,:,:3] / 255.
                img[mask==0,:] = 1.
            else:
                img = img[:,:,:3] / 255.                

            mem = memory()
            if mem['free'] < 0.1 * mem['total']:
                self.cache_img[self.datapath[index]] = (img, mask)

        return img, mask, cam

    def get_mesh(self, index):

        modelname = self.datapath[index].split('/')[-3]

        if modelname in self.cache_mesh:
            mesh = self.cache_mesh[modelname]
            return mesh

        cat = self.datapath[index].split('/')[-4]
        h5path = os.path.join(self.h5folder, cat, modelname+'.h5')
        # tic = time.time()
        h5_f = h5py.File(h5path)  
        if self.texture == 0:  
            verts, tris = h5_f['verts'][:], h5_f['tris'][:]
            if self.normalize:
                verts = pc_normalize(verts)
            mesh = (verts, tris)
        else:
            verts, tris, textures = h5_f['verts'][:], h5_f['tris'][:], h5_f['textures'][:]
            if self.normalize:
                verts = pc_normalize(verts)
            mesh = (verts, tris, textures)

        h5_f.close()
        # print 'mesh', time.time() - tic

        mem = memory()
        if mem['free'] < 0.1 * mem['total']:
            self.cache_mesh[modelname] = mesh

        return mesh

    def get_batch(self, current, withimg=True):
        bsize = self.batch_size

        if current + bsize > len(self.datapath):
            # random.shuffle(self.order)
            current = 0

        batch_verts = np.zeros((bsize, self.maxnverts, 3))
        batch_nverts = np.zeros((bsize, 1)).astype(np.int32)
        batch_tris = np.zeros((bsize, self.maxntris, 3)).astype(np.int32)
        batch_ntris = np.zeros((bsize, 1)).astype(np.int32)

        if withimg:
            batch_imgs = np.zeros((bsize, self.img_size[0], self.img_size[1], 3)).astype(np.float32)
            batch_cams = np.zeros((bsize, 9)).astype(np.int32)

        for i, i_d in enumerate(self.order[current:current+bsize]):
            mesh_data = self.get_mesh(i_d)
            if withimg:
                img, _, cam = self.get_img(i_d)

            batch_verts[i,:len(mesh_data[0]),:] = mesh_data[0]
            batch_tris[i,:len(mesh_data[1]),:] = mesh_data[1]
            batch_nverts[i,0] = len(mesh_data[0])
            batch_ntris[i,0] = len(mesh_data[1])
            if withimg:
                batch_imgs[i, :, :, :] = img[:,:,:3]
                if cam is not None:
                    batch_cams[i, :] = cam

        batch_data = {}
        batch_data['verts'] = batch_verts
        batch_data['nverts'] = batch_nverts
        batch_data['tris'] = batch_tris
        batch_data['ntris'] = batch_ntris

        if withimg:
            batch_data['imgs'] = batch_imgs
            batch_data['cams'] = batch_cams

        return batch_data


    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    path = '/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/ShapeNetCore.v2.render_img/03001627/a06c400e070343cfd8a56a98f4d239c3/image/r_000.png'
    img = np.asarray(Image.open(path), dtype=np.float32)
    mask = img[:,:,3]
    img = img[:,:,:3] / 255.
    img[mask==0,:] = 1.
    cv2.imwrite('tmp.png', (img * 255).astype(np.uint8)) 



