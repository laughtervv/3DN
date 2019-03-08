import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR))
try:
    import models.tf_ops.mesh_sampling.tf_meshsampling as tf_meshsampling
except:
    import models.tf_ops_server.mesh_sampling.tf_meshsampling as tf_meshsampling


def mesh_sample(src_verts, src_nverts, src_tris, src_ntris, batch_size, num_point, feats=None, scope='', random=None):

    with tf.variable_scope(scope) as sc:
        if random is not None:
            r1, r2, r = random
            # r1 = r1.initialized_value()
            # r2 = r2.initialized_value()
            # r = r.initialized_value()
        else:
            r1 = tf.random_uniform([batch_size, num_point], dtype=tf.float32)
            r2 = tf.random_uniform([batch_size, num_point], dtype=tf.float32)
            r = tf.random_uniform([batch_size, num_point], dtype=tf.float32)
        # print (scope,'r', r.get_shape())
        # print (scope,'feats', feats.get_shape())

        if feats == None:
            src_pc, src_pc_feats_from_verts, correspondingface = tf_meshsampling.mesh_sampling(src_verts, src_nverts, src_tris, src_ntris, src_verts, r, r1, r2)
        else:
            src_pc, src_pc_feats_from_verts, correspondingface = tf_meshsampling.mesh_sampling(src_verts, src_nverts, src_tris, src_ntris, feats, r, r1, r2)

        src_pc_feats_from_verts = tf.expand_dims(src_pc_feats_from_verts, 2)
        correspondingface = tf.expand_dims(correspondingface, 2)
        # print (scope,'src_pc_feats_from_verts', src_pc_feats_from_verts.get_shape())

    return src_pc, src_pc_feats_from_verts, correspondingface, r1, r2, r