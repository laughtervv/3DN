""" TF model for point cloud autoencoder. PointNet encoder, FC decoder.import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR))
import models.pointnet.model as pointnet
from models.ffd import *
try:
    import models.tf_ops.approxmatch.tf_approxmatch as tf_approxmatch
    import models.tf_ops.nn_distance.tf_nndistance as tf_nndistance
except:
    import models.tf_ops_server.approxmatch.tf_approxmatch as tf_approxmatch
    import models.tf_ops_server.nn_distance.tf_nndistance as tf_nndistance
Using GPU Chamfer's distance loss.

Author: Charles R. Qi
Date: May 2018
"""
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR))
import tf_util
# sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/nn_distance'))
# import tf_nndistance

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, scope='', num_point=None, bn_decay=None, ifglobal=False, bn=True):
    """ Autoencoder for point clouds.
    Input:
        point_cloud: TF tensor BxNx3
        is_training: boolean
        bn_decay: float between 0 and 1
    Output:
        net: TF tensor BxNx3, reconstructed point clouds
        end_points: dict
    """
    batch_size = point_cloud.get_shape()[0].value
    if num_point is None:
        num_point = point_cloud.get_shape()[1].value
    point_dim = point_cloud.get_shape()[2].value
    end_points = {}

    with tf.variable_scope(scope) as sc:

        input_image = tf.expand_dims(point_cloud, -1)

        # Encoder
        net = tf_util.conv2d(input_image, 64, [1,point_dim],
                             padding='VALID', stride=[1,1],
                             bn=bn, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        print('pointnet shape',  net.get_shape())
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=bn, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        point_feat = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=bn, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(point_feat, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=bn, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        pointwise = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=bn, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        # global_feat = tf_util.max_pool2d(pointwise, [num_point,1],
        #                                  padding='VALID', scope='maxpool')

        global_feat = tf.reduce_max(pointwise, axis = 1, keep_dims=True)

        print('maxpoolglobal_feat', global_feat.get_shape())

        feat0 = tf.reshape(global_feat, [batch_size, 1024])

        # FC Decoder
        net = None
        # if ifglobal:
        feat = tf_util.fully_connected(feat0, 1024, bn=bn, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        feat = tf_util.fully_connected(feat, 1024, bn=bn, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        # if num_point is not None:
        #     net = tf_util.fully_connected(feat, num_point * 3, activation_fn=None, scope='fc3')
        #     net = tf.reshape(net, (batch_size, num_point, 3))
        #     end_points['pc'] = net
    end_points['embedding'] = feat
    end_points['pointwise'] = pointwise

    return net, end_points

# def get_loss(pred, label, end_points):
#     """ pred: BxNx3,
#         label: BxNx3, """
#     dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pred, label)
#     loss = tf.reduce_mean(dists_forward+dists_backward)
#     end_points['pcloss'] = loss
#     return loss*100, end_points

def get_decoder(embedding, is_training, scope='', bn_decay=None, bn=True):

    batch_size = embedding.get_shape()[0].value
    print(embedding.shape)
    net = tf_util.fully_connected(embedding, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 1024*3, activation_fn=None, scope='fc3')
    pc_fc = tf.reshape(net, (batch_size, -1, 3))

    # UPCONV Decoder
    net = tf.reshape(embedding, [batch_size, 1, 1, -1])
    net = tf_util.conv2d_transpose(net, 512, kernel_size=[2,2], stride=[1,1], padding='VALID', scope='upconv1', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 256, kernel_size=[3,3], stride=[1,1], padding='VALID', scope='upconv2', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 256, kernel_size=[4,4], stride=[2,2], padding='VALID', scope='upconv3', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 128, kernel_size=[5,5], stride=[3,3], padding='VALID', scope='upconv4', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 3, kernel_size=[1,1], stride=[1,1], padding='VALID', scope='upconv5', activation_fn=None)

    pc_upconv = tf.reshape(net, [batch_size, -1, 3])

    # Set union
    reconst_pc = tf.concat(values=[pc_fc,pc_upconv], axis=1)
    
    return reconst_pc

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
        loss = get_loss(outputs[0], tf.zeros((32,1024,3)), outputs[1])
