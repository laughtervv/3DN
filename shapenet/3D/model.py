import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import models.pointnet.model as pointnet
import losses
import meshnet
import deformnet

def mesh_placeholder_inputs(batch_size, maxnverts, maxntris, num_points, scope=''):
    with tf.variable_scope(scope) as sc:
        # if batch_size == 1:
        #     verts_pl = tf.placeholder(tf.float32, shape=(batch_size, None, 3))
        #     nverts_pl = tf.placeholder(tf.int32, shape=(batch_size, 1))
        #     tris_pl = tf.placeholder(tf.int32, shape=(batch_size, None, 3))
        #     ntris_pl = tf.placeholder(tf.int32, shape=(batch_size, 1))
        # else:
            verts_pl = tf.placeholder(tf.float32, shape=(batch_size, maxnverts, 3))
            nverts_pl = tf.placeholder(tf.int32, shape=(batch_size, 1))
            tris_pl = tf.placeholder(tf.int32, shape=(batch_size, maxntris, 3))
            ntris_pl = tf.placeholder(tf.int32, shape=(batch_size, 1))

    mesh = {}

    mesh['verts'] = verts_pl
    mesh['nverts'] = nverts_pl
    mesh['tris'] = tris_pl
    mesh['ntris'] = ntris_pl

    return mesh

def get_model(src_mesh,
              ref_mesh, num_point, is_training, bn=False, bn_decay=None, localloss=True):

    src_verts = src_mesh['verts']
    src_nverts = src_mesh['nverts']
    src_tris = src_mesh['tris']
    src_ntris = src_mesh['ntris']

    ref_verts = ref_mesh['verts']
    ref_nverts = ref_mesh['nverts']
    ref_tris = ref_mesh['tris']
    ref_ntris = ref_mesh['ntris']
    # ref_pc = ref_mesh['solidbinvoxpc']

    batch_size = src_verts.get_shape()[0].value
    num_src_verts = src_verts.get_shape()[1].value

    end_points = {}
    end_points['src_mesh'] = src_mesh
    end_points['ref_mesh'] = ref_mesh

    # source
    src_pc, src_pc_, correpondingface, _, _, _ = meshnet.mesh_sample(src_verts, src_nverts, src_tris, src_ntris, batch_size, num_point, src_verts, scope='meshsample') 
    end_points['src_pc'] = src_pc

    _, src_feats = pointnet.get_model(src_pc, is_training, num_point=num_point, scope='srcpc', bn=bn, bn_decay=bn_decay)
    end_points['src_feats'] = src_feats

    ref_pc, _, correpondingface, _, _, _ = meshnet.mesh_sample(ref_verts, ref_nverts, ref_tris, ref_ntris, batch_size, num_point, ref_verts, scope='meshsample') 
    end_points['ref_pc'] = ref_pc
    _, ref_feats = pointnet.get_model(ref_pc, is_training, num_point=num_point, scope='refpc', bn=bn, bn_decay=bn_decay)
    end_points['ref_feats'] = ref_feats

    with tf.variable_scope("sharebiasnet") as scope:     
        pred_pc, centroids = deformnet.get_pred_foldenet_basic(src_pc, src_feats['embedding'], ref_feats['embedding'], is_training, batch_size, num_point, bn, bn_decay)
        end_points['pred_pc'] = pred_pc

        scope.reuse_variables() 
        pred_verts, _ = deformnet.get_pred_foldenet_basic(src_verts, src_feats['embedding'], ref_feats['embedding'], is_training, batch_size, num_point, bn, bn_decay)
        end_points['pred_verts'] = pred_verts

        if localloss:
            delta = 0.005
            localpclap_pred_pc = [pred_pc]

            src_pc_x = src_pc[:,:,0]
            src_pc_y = src_pc[:,:,1]
            src_pc_z = src_pc[:,:,2]

            src_pc_x1 = src_pc_x + delta
            src_pc_x1 = tf.concat(axis=2, values=[tf.expand_dims(src_pc_x1, -1), src_pc[:,:,1:]])
            src_pc_x2 = src_pc_x - delta
            src_pc_x2 = tf.concat(axis=2, values=[tf.expand_dims(src_pc_x2, -1), src_pc[:,:,1:]])

            src_pc_y1 = src_pc_y + delta
            src_pc_y1 = tf.concat(axis=2, values=[tf.expand_dims(src_pc[:,:,0], -1), tf.expand_dims(src_pc_y1, -1), tf.expand_dims(src_pc[:,:,2], -1)])
            src_pc_y2 = src_pc_y - delta
            src_pc_y2 = tf.concat(axis=2, values=[tf.expand_dims(src_pc[:,:,0], -1), tf.expand_dims(src_pc_y2, -1), tf.expand_dims(src_pc[:,:,2], -1)])

            src_pc_z1 = src_pc_z + delta
            src_pc_z1 = tf.concat(axis=2, values=[src_pc[:,:,:2], tf.expand_dims(src_pc_z1, -1)])
            src_pc_z2 = src_pc_z - delta
            src_pc_z2 = tf.concat(axis=2, values=[src_pc[:,:,:2], tf.expand_dims(src_pc_z2, -1)])      

            pred_pc_x1, _ = deformnet.get_pred_foldenet_basic(src_pc_x1, src_feats['embedding'], ref_feats['embedding'], is_training, batch_size, num_point, bn, bn_decay)
            pred_pc_x2, _ = deformnet.get_pred_foldenet_basic(src_pc_x2, src_feats['embedding'], ref_feats['embedding'], is_training, batch_size, num_point, bn, bn_decay)

            pred_pc_y1, _ = deformnet.get_pred_foldenet_basic(src_pc_y1, src_feats['embedding'], ref_feats['embedding'], is_training, batch_size, num_point, bn, bn_decay)
            pred_pc_y2, _ = deformnet.get_pred_foldenet_basic(src_pc_y2, src_feats['embedding'], ref_feats['embedding'], is_training, batch_size, num_point, bn, bn_decay)

            pred_pc_z1, _ = deformnet.get_pred_foldenet_basic(src_pc_z1, src_feats['embedding'], ref_feats['embedding'], is_training, batch_size, num_point, bn, bn_decay)
            pred_pc_z2, _ = deformnet.get_pred_foldenet_basic(src_pc_z2, src_feats['embedding'], ref_feats['embedding'], is_training, batch_size, num_point, bn, bn_decay)

            localpclap_pred_pc += [[pred_pc_x1, pred_pc_x2],
                                   [pred_pc_y1, pred_pc_y2], 
                                   [pred_pc_z1, pred_pc_z2]]
            end_points['localpclap_pred_pc'] = localpclap_pred_pc

    return end_points



def get_loss(end_points, num_class=4):

    end_points['losses'] = {}

    ## point cloud loss
    pred_pc = end_points['pred_pc']
    ref_pc = end_points['ref_pc']
    pc_cf_loss, end_points = losses.get_chamfer_loss(pred_pc, ref_pc, end_points)
    pc_cf_loss = 10000 * pc_cf_loss
    
    pc_em_loss, end_points = losses.get_em_loss(pred_pc, ref_pc, end_points)
    end_points['losses']['pc_cf_loss'] = pc_cf_loss
    end_points['losses']['pc_em_loss'] = pc_em_loss

    ## mesh loss
    pred_verts = end_points['pred_verts']
    src_mesh = end_points['src_mesh']
    src_verts = src_mesh['verts']
    src_nverts = src_mesh['nverts']
    src_tris = src_mesh['tris']
    src_ntris = src_mesh['ntris']
    batch_size = src_verts.get_shape()[0].value
    num_point = ref_pc.get_shape()[1].value
    _, pred_pc_fromverts, correpondingface, _, _, _ = meshnet.mesh_sample(src_verts, src_nverts, src_tris, src_ntris, batch_size, num_point, src_verts, scope='meshsample') 
    pred_pc_fromverts = tf.squeeze(pred_pc_fromverts, axis=2)
    mesh_cf_loss, end_points = losses.get_chamfer_loss(pred_pc_fromverts, ref_pc, end_points)
    mesh_cf_loss = 1000 * mesh_cf_loss
    
    mesh_em_loss, end_points = losses.get_em_loss(pred_pc_fromverts, ref_pc, end_points)
    end_points['losses']['mesh_cf_loss'] = mesh_cf_loss
    end_points['losses']['mesh_em_loss'] = mesh_em_loss

    ## symmetry loss
    pred_pc_xflip = tf.concat([tf.expand_dims(-pred_pc[:,:,0], axis=2), tf.expand_dims(pred_pc[:,:,1], axis=2), tf.expand_dims(pred_pc[:,:,2], axis=2)], axis = 2)
    pc_symmetry_loss, end_points = losses.get_chamfer_loss(pred_pc_xflip, ref_pc, end_points)
    pc_symmetry_loss = 1000 * pc_symmetry_loss
    match_symmetry_loss, end_points = losses.get_em_loss(pred_pc_xflip, ref_pc, end_points)
    end_points['losses']['pc_symmetry_loss'] = pc_symmetry_loss
    end_points['losses']['match_symmetry_loss'] = match_symmetry_loss

    ## local permutation invariance loss
    localpclap_pred_pc = end_points['localpclap_pred_pc']
    pc_local_laplacian_loss, end_points = losses.get_pc_local_laplacian_loss(localpclap_pred_pc, end_points=end_points)
    pc_local_laplacian_loss = 1000 * pc_local_laplacian_loss
    end_points['losses']['pc_local_laplacian_loss'] = pc_local_laplacian_loss

    ## mesh laplacian loss
    mesh_laplacian_loss, _ = losses.get_laplacian_loss(src_mesh, pred_verts)
    mesh_laplacian_loss = 0.01 * mesh_laplacian_loss
    end_points['losses']['mesh_laplacian_loss'] = mesh_laplacian_loss

    loss = pc_cf_loss + pc_em_loss + \
           pc_symmetry_loss + match_symmetry_loss + \
           pc_local_laplacian_loss + \
           mesh_laplacian_loss + \
           mesh_cf_loss + mesh_em_loss

    end_points['losses']['overall_loss'] = loss

    for lossname in end_points['losses'].keys():
        tf.summary.scalar(lossname, end_points['losses'][lossname])

    return loss, end_points