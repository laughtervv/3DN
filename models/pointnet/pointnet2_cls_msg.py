import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg
from pointnet_util import pointnet_sa_module, pointnet_fp_module


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, scope='', num_point=None, bn_decay=None, ifglobal=False, bn=True, end_points={}):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    l0_xyz = point_cloud
    l0_points = None

    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, 512, [0.1,0.2,0.4], [16,32,128], [[32,32,64], [64,64,128], [64,96,128]], is_training, bn_decay, scope='layer1', use_nchw=True, bn = bn)
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, 128, [0.2,0.4,0.8], [32,64,128], [[64,64,128], [128,128,256], [128,128,256]], is_training, bn_decay, scope='layer2', bn = bn)
    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3', bn = bn)

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=bn, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['embedding'] = net
    # net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp1')
    # net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    # net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp2')
    # net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
    end_points['l2_xyz'] = l2_xyz
    end_points['l3_xyz'] = l3_xyz
    end_points['l1_xyz'] = l1_xyz
    end_points['l0_xyz'] = l0_xyz
    end_points['l2_points'] = l2_points
    end_points['l3_points'] = l3_points
    end_points['l1_points'] = l1_points
    end_points['l0_points'] = l0_points

    return net, end_points

def get_decoder(embedding, is_training, scope='pointnet2_decoder', bn_decay=None, bn=True, end_points = {}):
    with tf.name_scope(scope) as sc:
        l2_xyz = end_points['l2_xyz'] 
        l3_xyz = end_points['l3_xyz'] 
        l1_xyz = end_points['l1_xyz'] 
        l0_xyz = end_points['l0_xyz'] 
        l2_points = end_points['l2_points'] 
        l3_points = end_points['l3_points'] 
        l1_points = end_points['l1_points'] 
        l0_points = end_points['l0_points'] 

        batch_size = embedding.get_shape()[0].value
        # net = tf_util.fully_connected(embedding, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        # net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        # net = tf_util.fully_connected(net, 1024*3, activation_fn=None, scope='fc3')
        # pc_fc = tf.reshape(net, (batch_size, -1, 3))

        embedding = tf.expand_dims(embedding, axis=1)
        l3_points = tf.concat([embedding, l3_points], axis = -1)

        # Feature Propagation layers
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer1')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer2')
        l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer3')

        # FC layers
        net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='decoder_fc1', bn_decay=bn_decay)
        net = tf_util.conv1d(l0_points, 3, 1, padding='VALID', bn=False, is_training=is_training, scope='decoder_fc2', bn_decay=None, activation_fn=None)
        # net = tf_util.conv2d_transpose(net, 3, kernel_size=[1,1], stride=[1,1], padding='VALID', scope='fc2', activation_fn=None)

        reconst_pc = tf.reshape(net, [batch_size, -1, 3])

    
    return reconst_pc

def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
