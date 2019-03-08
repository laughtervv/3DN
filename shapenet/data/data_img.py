'''
    Dataset for shapenet part segmentaion.
'''

import os
import json
import numpy as np
import sys
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
import h5py
import cv2


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

class ImgDataset_test():
    def __init__(self, listfile):

        self.inputlistfile = listfile

        if isinstance(self.inputlistfile, str):
            with open(self.inputlistfile, 'r') as f:
                lines = f.read().splitlines()
                self.datapath = [line.strip() for line in lines]
        else:
            self.datapath = listfile

        self.order = list(range(len(self.datapath)))

    def __getitem__(self, index):

        batch_img = np.ones([1, 600, 600, 3], dtype=np.float32)
        img = Image.open(self.datapath[index])#

        if img.size[0] > img.size[1]:
            img.thumbnail((400, int(400.*img.size[0]/img.size[1])), Image.ANTIALIAS)
        else:
            img.thumbnail((int(400.*img.size[1]/img.size[0]), 400), Image.ANTIALIAS)            

        img = np.asarray(img, dtype=np.float32)
        print(img.shape)
        if len(img.shape) == 2:
            img = np.transpose(np.asarray([img,img,img]), (1,2,0))

        if img.shape[2] == 4:
            mask = img[:,:,3]
            img = img[:,:,:3] / 255.
            img[mask==0,:] = 1.
        else:
            img /= 255.
        offsetx = int((600. - img.shape[0]) / 2)
        offsety = int((600. - img.shape[1]) / 2)
        batch_img[:,offsetx:offsetx+img.shape[0],offsety:offsety+img.shape[1], :] = img

        return batch_img


    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    path = '/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/ShapeNetCore.v2.render_img/03001627/a06c400e070343cfd8a56a98f4d239c3/image/r_000.png'
    img = np.asarray(Image.open(path), dtype=np.float32)
    mask = img[:,:,3]
    img = img[:,:,:3] / 255.
    img[mask==0,:] = 1.
    cv2.imwrite('tmp.png', (img * 255).astype(np.uint8)) 



