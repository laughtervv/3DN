import tensorflow as tf
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module

def displacement_decoder(net, batch_size, is_training, bn=True, bn_decay=None, scope = '', outputdim=3):

    net2 = tf_util.conv2d(net, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, scope=scope+'decoder/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, scope=scope+'decoder/conv2')
    net2 = tf_util.conv2d(net2, outputdim, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, scope=scope+'decoder/conv5')

    net2 = tf.reshape(net2, [batch_size, -1, outputdim])
    return net2

def get_pred_foldenet_basic(src_pc, src_feats, ref_feats, is_training, batch_size, num_point, bn, bn_decay):
    ##TODO: Symmetry

    globalfeats = tf.concat([src_feats, ref_feats], axis=1)

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])

    concat = tf.concat(axis=3, values=[tf.expand_dims(src_pc,2), globalfeats_expand])
    displacement = displacement_decoder(concat, batch_size, is_training, bn=bn, bn_decay=bn_decay)

    concat = tf.concat(axis=3, values=[tf.expand_dims(displacement,2), globalfeats_expand])
    displacement = displacement_decoder(concat, batch_size, is_training, bn=bn, bn_decay=bn_decay, scope='fold2/')

    concat2 = tf.concat(axis=3, values=[tf.expand_dims(displacement,2), concat])
    displacement = displacement_decoder(concat2, batch_size, is_training, bn=bn, bn_decay=bn_decay, scope='fold3/')

    pred = src_pc + displacement

    return pred, displacement

