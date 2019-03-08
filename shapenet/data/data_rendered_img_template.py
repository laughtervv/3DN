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
    def __init__(self, listfile, batch_size=8, img_size=(600,600), maxnverts=10000, maxntris=10000, normalize = True, h5folder='/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/ShapeNetCore.v2.h5'):

        self.maxnverts = maxnverts
        self.maxntris = maxntris
        self.inputlistfile = listfile
        self.normalize = normalize
        self.h5folder = h5folder
        self.img_size = img_size
        self.batch_size = batch_size

        with open(self.inputlistfile, 'r') as f:
            lines = f.read().splitlines()
            self.datapath_src = [line.strip().split(' ')[1] for line in lines]
            self.datapath_img = [line.strip().split(' ')[0] for line in lines]

        self.data_num = len(lines)
        self.order = list(range(self.data_num))

        self.cache_mesh = {}
        self.cache_img = {}

    def get_batch(self, index):
        bsize = self.batch_size

        src_batch_verts = np.zeros((bsize, self.maxnverts, 3))
        src_batch_nverts = np.zeros((bsize, 1)).astype(np.int32)
        src_batch_tris = np.zeros((bsize, self.maxntris, 3)).astype(np.int32)
        src_batch_ntris = np.zeros((bsize, 1)).astype(np.int32)

        tgt_batch_verts = np.zeros((bsize, self.maxnverts, 3))
        tgt_batch_nverts = np.zeros((bsize, 1)).astype(np.int32)
        tgt_batch_tris = np.zeros((bsize, self.maxntris, 3)).astype(np.int32)
        tgt_batch_ntris = np.zeros((bsize, 1)).astype(np.int32)
        tgt_batch_imgs = np.zeros((bsize, self.img_size[0], self.img_size[1], 3)).astype(np.float32)
        
        cnt = 0
        i = index
        while cnt < self.batch_size:
            tgt_mesh_data = self.get_mesh(self.datapath_img[self.order[i]])
            img, _ = self.get_img(self.datapath_img[self.order[i]])
            src_mesh_data = self.get_mesh(self.datapath_src[self.order[i]], False)

            i += 1
            if i >= self.data_num:
                i = 0

            if tgt_mesh_data == None or src_mesh_data == None:
                continue

            src_batch_verts[cnt,:len(src_mesh_data[0]),:] = src_mesh_data[0]
            src_batch_tris[cnt,:len(src_mesh_data[1]),:] = src_mesh_data[1]
            src_batch_nverts[cnt,0] = len(src_mesh_data[0])
            src_batch_ntris[cnt,0] = len(src_mesh_data[1])

            tgt_batch_verts[cnt,:len(tgt_mesh_data[0]),:] = tgt_mesh_data[0]
            tgt_batch_tris[cnt,:len(tgt_mesh_data[1]),:] = tgt_mesh_data[1]
            tgt_batch_nverts[cnt,0] = len(tgt_mesh_data[0])
            tgt_batch_ntris[cnt,0] = len(tgt_mesh_data[1])
            tgt_batch_imgs[cnt, :, :, :] = img#[:,:,:3]
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
        tgt_batch_data['imgs'] = tgt_batch_imgs

        return src_batch_data, tgt_batch_data

    def get_img(self, path):

        if path in self.cache_img:
            img, mask = self.cache_img[path]
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if self.img_size[0] != img.shape[0] or self.img_size[1] != img.shape[1]:
                print img.shape
                img = cv2.resize(img, self.img_size).astype(np.float32)

            mask = np.zeros([img.shape[0], img.shape[1]])
            if img.shape[2] == 4:
                mask = img[:,:,3]
                img = img[:,:,:3] / 255.
                img[mask==0,:] = 1.
            else:
                img = img[:,:,:3] / 255.  
                mask[(img[:,:,0]!=0)|(img[:,:,1]!=0)|(img[:,:,2]!=0),:] = 1.              

            mem = memory()
            if mem['free'] < 0.1 * mem['total']:
                self.cache_img[path] = (img, mask)

        return img, mask

    def get_mesh(self, path, imgpath=True):

        if imgpath:
            modelname = path.split('/')[-3]
            cat = path.split('/')[-4]
            h5path = os.path.join(self.h5folder, cat, modelname+'.h5')
        else:
            h5path = path

        if h5path in self.cache_mesh:
            mesh = self.cache_mesh[h5path]
            return mesh

        try:
            h5_f = h5py.File(h5path)
            if (self.maxnverts != -1 and h5_f['verts'].shape[0] > self.maxnverts) or (self.maxntris != -1 and h5_f['tris'].shape[0] > self.maxntris):
                raise Exception()

            verts, tris = h5_f['verts'][:], h5_f['tris'][:]
        except:
            h5_f.close()
            self.cache_mesh[h5path] = None
            return None

        if self.normalize:
            verts = pc_normalize(verts)

        h5_f.close()

        mem = memory()
        if mem['free'] < 0.1 * mem['total']:
            self.cache_mesh[h5path] = (verts, tris)

        return verts, tris

    def __len__(self):
        return self.data_num

if __name__ == '__main__':
    MAX_NVERTS = MAX_NTRIS = 5000
    d = RenderedImgDataset('/media/hdd2/data/ShapeNet/filelists/ShapeNetCore.v2.h5/03001627_obj_5000.lst', maxnverts=MAX_NVERTS, maxntris=MAX_NTRIS)

    batch_size = 20
    tic = time.time()
    for i in range(10):
        i = i * batch_size
        batch = d.get_batch(i)
    print (time.time() - tic)

