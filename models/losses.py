import tensorflow as tf
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR))
import models.pointnet.model_fc_upconv as pointnet_fc_upconv
import models.pointnet.model_seg as model_seg
try:
    import models.tf_ops.approxmatch.tf_approxmatch as tf_approxmatch
    import models.tf_ops.nn_distance.tf_nndistance as tf_nndistance
    import models.tf_ops.mesh_laplacian.tf_meshlaplacian as tf_meshlaplacian
    # import models.tf_ops.neural_renderer.tf_neuralrenderer as tf_neuralrenderer
except:
    import models.tf_ops_server.approxmatch.tf_approxmatch as tf_approxmatch
    import models.tf_ops_server.nn_distance.tf_nndistance as tf_nndistance
    import models.tf_ops_server.mesh_laplacian.tf_meshlaplacian as tf_meshlaplacian
# from models.tf_ops.grouping.tf_grouping import query_ball_point, group_point
from pyquaternion import Quaternion


def get_repulsion_loss4(pred, nsample=20, radius=0.07, end_points={}):
    #### repulsion loss: https://github.com/yulequan/PU-Net/blob/master/code/model_utils.py
    # pred: (batch_size, npoint,3)
    idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    ##get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = tf.maximum(1e-12,dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square/h**2)
    uniform_loss = tf.reduce_mean(radius-dist*weight)

    return uniform_loss, end_points


def get_chamfer_loss(pred_pc, ref_pc, end_points={}):

    dists_predpc, _, dists_refpc, _ = tf_nndistance.nn_distance(pred_pc, ref_pc)
    pc_loss = (tf.reduce_mean(dists_predpc)+tf.reduce_mean(dists_refpc))

    return pc_loss, end_points


def get_em_loss(pred_pc, ref_pc, end_points={}):

    match = tf_approxmatch.approx_match(pred_pc, ref_pc)
    match_loss = tf.reduce_mean(tf_approxmatch.match_cost(pred_pc, ref_pc, match))

    return match_loss, end_points

def get_pcloss_withseg(src_pc, pred_pc, ref_pc, num_class, end_points={}):

    batch_size = pred_pc.shape[0]
    with tf.variable_scope("part_seg") as scope:    
        if ('pred_label' not in end_points.keys()):
            pred_label, _ = model_seg.get_model(src_pc, tf.constant(False), None)
            pred_label = tf.argmax(pred_label, axis = 2)#tf.expand_dims(, axis = -1)
            end_points['pred_label'] = pred_label
        else:
            pred_label = end_points['pred_label']

        if ('ref_label' not in end_points.keys()):    
            scope.reuse_variables() 
            ref_label, _ = model_seg.get_model(ref_pc, tf.constant(False), None)
            ref_label = tf.argmax(ref_label, axis = 2)
            end_points['ref_label'] = ref_label
        else:
            ref_label = end_points['ref_label']


        pc_loss = tf.constant(0.)
        match_loss = tf.constant(0.)


    for i_c in range(num_class):#
        i_c_tensor = tf.constant(i_c, dtype = tf.int64)
        predpc_part_mask = tf.equal(pred_label, i_c_tensor)
        refpc_part_mask = tf.equal(ref_label, i_c_tensor)

        #--- losses
        # predpc_part_mask = tf.tile(predpc_part_mask, [1, 1, 3])
        # refpc_part_mask = tf.tile(refpc_part_mask, [1, 1, 3])
        print('predpc_part_mask', predpc_part_mask, 'pred_pc', pred_pc)

        # the reason for processing each individual batch is the part point cloud size are not the same across batches
        for i_b in range(batch_size):

            masked_pred_pc = tf.reshape(tf.boolean_mask(pred_pc[i_b,:,:], predpc_part_mask[i_b,:]), [1, -1, 3])
            masked_ref_pc = tf.reshape(tf.boolean_mask(ref_pc[i_b,:,:], refpc_part_mask[i_b,:]), [1, -1, 3])


            def chamfer(): 
                dists_predpc, _, dists_refpc, _ = tf_nndistance.nn_distance(masked_pred_pc, masked_ref_pc)
                return tf.reduce_mean(dists_predpc) + tf.reduce_mean(dists_refpc)

            def em(): 
                match = tf_approxmatch.approx_match(masked_ref_pc, masked_pred_pc)
                return tf.reduce_mean(tf_approxmatch.match_cost(masked_ref_pc, masked_pred_pc, match))
            # pc_loss_ =  #* tf.cast(tf.greater(tf.shape(masked_pred_pc)[1], tf.constant(0)), tf.float32)  * tf.cast(tf.greater(tf.shape(masked_ref_pc)[1], tf.constant(0)), tf.float32)
            pc_loss_ = tf.cond(tf.logical_and(tf.shape(masked_ref_pc)[1] > 10, tf.shape(masked_pred_pc)[1] > 10), chamfer, lambda: tf.constant(0.))
            pc_loss += pc_loss_

            # end_points['losses'][pc_loss_]
            
            match_loss_ = tf.cond(tf.logical_and(tf.shape(masked_ref_pc)[1] > 10, tf.shape(masked_pred_pc)[1] > 10), em, lambda: tf.constant(0.))
            match_loss += match_loss_
            # end_points['masked_ref_pc'] = masked_ref_pc
            # end_points['masked_ref_pc'] = masked_ref_pc

    match_loss /= num_class#tf.reduce_mean(match_loss)
    pc_loss /= num_class#tf.reduce_mean(pc_loss)

    return pc_loss, match_loss, end_points


def get_segmentation_loss(src_pc, pred_pc, num_class, end_points={}):

    batch_size = pred_pc.shape[0]

    with tf.variable_scope("part_seg") as scope:     
        src_label, _ = model_seg.get_model(src_pc, tf.constant(False), None)
        src_label = tf.argmax(src_label, axis = 2)

        scope.reuse_variables() 
        pred_label, _ = model_seg.get_model(pred_pc, tf.constant(False), None)

        print ('src_label', src_label.shape, 'pred_label', pred_label.shape)
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_label, labels=src_label)
    loss = tf.reduce_mean(loss)

    pred_label = tf.argmax(pred_label, axis = 2)

    end_points['pred_label'] = pred_label
    end_points['ref_label'] = src_label


    return loss, end_points

def get_laplacian_loss(src_mesh, pred_verts, end_points={}):

    src_verts = src_mesh['verts']
    src_nverts = src_mesh['nverts']
    src_tris = src_mesh['tris']
    src_ntris = src_mesh['ntris']
    laplacian1, _, _ = tf_meshlaplacian.mesh_laplacian(src_verts, src_nverts, src_tris, src_ntris)
    laplacian2, _, _ = tf_meshlaplacian.mesh_laplacian(pred_verts, src_nverts, src_tris, src_ntris)
    laplacian_loss = (laplacian1 - laplacian2) #* tf.cast(src_vert_part, tf.float32)
    laplacian_loss = tf.reduce_mean(tf.nn.l2_loss(laplacian_loss))

    return laplacian_loss, end_points

def get_origin_laplacian_loss(src_mesh, pred_verts, end_points={}):

    src_verts = src_mesh['origin_verts']
    src_nverts = src_mesh['nverts']
    src_tris = src_mesh['tris']
    src_ntris = src_mesh['ntris']
    laplacian1, _, _ = tf_meshlaplacian.mesh_laplacian(src_verts, src_nverts, src_tris, src_ntris)
    laplacian2, _, _ = tf_meshlaplacian.mesh_laplacian(pred_verts, src_nverts, src_tris, src_ntris)
    laplacian_loss = (laplacian1 - laplacian2) #* tf.cast(src_vert_part, tf.float32)
    laplacian_loss = tf.reduce_mean(tf.nn.l2_loss(laplacian_loss))

    return laplacian_loss, end_points


def get_pc_smooth_loss(pred_pc, end_points={}):
    """
        # pred_pc: (list: 4) [(x,y,z), [(x+2,y,z), (x+1,y,z), (x-1,y,z)], 
        #                              [(x,y+2,z), (x,y+1,z), (x,y-1,z)], 
        #                              [(x,y,z+2), (x,y,z+1), (x,y,z-1)]]

        # x consistancy: (x+2,y,z)+3(x,y,z)-3(x+1,y,z)-(x-1,y,z)
        # y consistancy: (x,y+2,z)+3(x,y,z)-3(x,y+1,z)-(x,y-1,z)
        # z consistancy: (x,y,z+2)+3(x,y,z)-3(x,y,z+1)-(x,y,z-1)
        pred_pc: (list: 4) [(x,y,z), [(x+1,y,z), (x-1,y,z)], 
                                     [(x,y+1,z), (x,y-1,z)], 
                                     [(x,y,z+1), (x,y,z-1)]]
    """

    xyz = pred_pc[0]
    loss_ = 0
    for i in range(3):
        loss_ += tf.reduce_mean(tf.nn.l2_loss(pred_pc[i+1][0] + xyz * 3. - pred_pc[i+1][1] * 3. - pred_pc[i+1][2]))

    return loss_, end_points

def get_pc_local_laplacian_loss(pred_pc, end_points={}, hingeloss=True):
    """
        pred_pc: (list: 4) [(x,y,z), [(x+1,y,z), (x-1,y,z)], 
                                     [(x,y+1,z), (x,y-1,z)], 
                                     [(x,y,z+1), (x,y,z-1)]]
    """

    pred_xyz = pred_pc[0]
    loss_ = 0
    for i in range(3):
        pred_edge1 = pred_pc[i+1][0] - pred_xyz
        pred_edge2 = pred_xyz - pred_pc[i+1][1]

        loss_ += tf.reduce_mean(-tf.minimum(pred_edge1, 0))
        loss_ += tf.reduce_mean(-tf.minimum(pred_edge2, 0))
        # if hingeloss:
        #     margin = tf.constant(0.005)
        #     loss_ += tf.reduce_mean(tf.maximum(tf.subtract(margin, pred_edge1), 0))
        #     loss_ += tf.reduce_mean(tf.maximum(tf.subtract(margin, pred_edge2), 0))
        # else:
        #     loss_ += tf.reduce_mean(tf.nn.l2_loss(pred_edge1 - pred_edge2))

    end_points['debug'] = loss_
    return loss_, end_points

def get_pc_local_laplacian_loss_old(src_pc, pred_pc, cosine=False, end_points={}, delta=0.01):
    """
        pred_pc: (list: 4) [(x,y,z), [(x+1,y,z), (x-1,y,z)], 
                                     [(x,y+1,z), (x,y-1,z)], 
                                     [(x,y,z+1), (x,y,z-1)]]
    """

    pred_xyz = pred_pc[0]
    src_xyz = src_pc[0]
    loss_ = 0
    for i in range(3):
        pred_edge1 = pred_pc[i+1][0] - pred_xyz
        src_edge1 = src_pc[i+1][0] - src_xyz
        pred_edge2 = pred_xyz - pred_pc[i+1][1]
        src_edge2 = src_xyz - src_pc[i+1][1]

        print('src_edge1', src_edge1.get_shape(), 'pred_edge1', pred_edge1.get_shape())

        if cosine:
            dot = tf.reduce_sum(src_edge1 * pred_edge1, axis=2)
            src_norm = tf.norm(src_edge1, axis=2)
            pred_norm = tf.norm(pred_edge1, axis=2)
            eps = 1e-7
            cosine_ = dot / tf.maximum(src_norm * pred_norm, eps)

            print('dot', dot.get_shape(), 'src_norm', src_norm.get_shape(), 'pred_norm', pred_norm.get_shape())

            loss_ += tf.reduce_mean(1. - cosine_)#tf.reduce_mean(tf.losses.cosine_distance(src_edge1, pred_edge1, dim = 2))

            dot = tf.reduce_sum(src_edge2 * pred_edge2, axis=2)
            src_norm = tf.norm(src_edge2, axis=2)
            pred_norm = tf.norm(pred_edge2, axis=2)
            eps = 1e-7
            cosine_ = dot / tf.maximum(src_norm * pred_norm, eps)

            loss_ += tf.reduce_mean(1. - cosine_)#tf.reduce_mean(tf.losses.cosine_distance(src_edge2, pred_edge2, dim = 2))

            end_points['cosine'] = cosine_#tf.losses.cosine_distance(src_edge2, pred_edge2, dim = 2)

        else:
            loss_ += tf.reduce_mean(tf.nn.l2_loss(pred_edge1 - delta))
            loss_ += tf.reduce_mean(tf.nn.l2_loss(pred_edge2 - delta))

    end_points['debug'] = loss_
    return loss_, end_points

def get_densecorr_loss(src_pc, ref_pc, pred_pc, end_points={}):

    def get_densecorrepondence(src_pc, ref_pc):

        batch_size = src_pc.get_shape()[0].value
        _, src_endpoints = pointnet_fc_upconv.get_model(src_pc, is_training=False)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            _, ref_endpoints = pointnet_fc_upconv.get_model(ref_pc, is_training=False)
        
        src_feat = tf.squeeze(src_endpoints['pointwise'], axis=2)
        ref_feat = tf.squeeze(ref_endpoints['pointwise'], axis=2)

        r1 = tf.reduce_sum(src_feat * src_feat, 2)
        r1 = tf.reshape(r1, [batch_size, -1, 1])
        r2 = tf.reduce_sum(ref_feat * ref_feat, 2)
        r2 = tf.reshape(r2, [batch_size, -1, 1])

        D = r1 - 2 * tf.matmul(src_feat, tf.transpose(ref_feat, perm=[0, 2, 1])) + tf.transpose(r2, perm=[0, 2, 1])

        ind1 = tf.argmin(D, axis=1) # BxN
        ind2 = tf.argmin(D, axis=2)

        return ind1, ind2 #

    batch_size = src_pc.get_shape()[0].value
    num_point = src_pc.get_shape()[1].value

    src_ind, ref_ind = get_densecorrepondence(src_pc, ref_pc)
    src_ind = tf.expand_dims(src_ind, -1)# BxNx1
    ref_ind = tf.expand_dims(ref_ind, -1)

    B = np.zeros([batch_size, num_point, 1], dtype=np.int64)
    for iB in range(batch_size):
        B[iB,:,:] = iB
    B = tf.constant(B)

    # ref_correpondingface: B x num_point x 1
    refpc_indices = tf.concat(axis=2, values=[B, ref_ind])
    pred_pc_gt = tf.gather_nd(ref_pc, refpc_indices, name=None)

    srcpc_indices = tf.concat(axis=2, values=[B, src_ind])#BxNx2
    ref_pc_gt = tf.gather_nd(pred_pc, srcpc_indices, name=None)
    ref_pc_gt_forsource = tf.gather_nd(src_pc, srcpc_indices, name=None)

    pc_loss_src = tf.reduce_mean(tf.nn.l2_loss(pred_pc_gt - pred_pc))
    pc_loss_ref = tf.reduce_mean(tf.nn.l2_loss(ref_pc_gt - ref_pc))
    # end_points['pred_pc_gt'] = pred_pc_gt
    # end_points['ref_pc_gt'] = ref_pc_gt
    densecorr_loss = pc_loss_src + pc_loss_ref

    return densecorr_loss, end_points


def get_edgelen_loss(src_mesh, pred_verts, end_points={}, use_l2=False, margin=2.):
    ##Encourages dihedral angle to be 180 degrees.

    verts = src_mesh['verts']
    nverts = src_mesh['nverts']
    tris = src_mesh['tris'] # B x maxntris x 3
    ntris = src_mesh['ntris']
    vertsmask = src_mesh['vertsmask']
    trismask = tf.cast(tf.squeeze(src_mesh['trismask']), tf.bool)

    maxntris = src_mesh['tris'].get_shape()[1]
    batch_size = src_mesh['tris'].get_shape()[0]

    B = np.zeros([batch_size, maxntris, 1], dtype=np.int64)
    for iB in range(batch_size):
        B[iB,:,:] = iB
    B = tf.constant(B)
    tris = tf.cast(tris, tf.int64)

    print('tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 0], axis=-1)])', tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 0], axis=-1)]).shape)

    verts_A = tf.gather_nd(verts, tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 0], axis=-1)]))
    verts_B = tf.gather_nd(verts, tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 1], axis=-1)]))
    verts_C = tf.gather_nd(verts, tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 2], axis=-1)]))
    print ('verts_A', verts_A.shape)

    edgelenAB = tf.reduce_sum(tf.sqrt((verts_A - verts_B) * (verts_A - verts_B)), axis=2)
    edgelenBC = tf.reduce_sum(tf.sqrt((verts_C - verts_B) * (verts_C - verts_B)), axis=2)
    edgelenAC = tf.reduce_sum(tf.sqrt((verts_A - verts_C) * (verts_A - verts_C)), axis=2)
    end_points['debug'] = tf.boolean_mask(edgelenAB, trismask)

    verts_A_pred = tf.gather_nd(pred_verts, tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 0], axis=-1)]))
    verts_B_pred = tf.gather_nd(pred_verts, tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 1], axis=-1)]))
    verts_C_pred = tf.gather_nd(pred_verts, tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 2], axis=-1)]))

    edgelenAB_pred = tf.reduce_sum(tf.sqrt((verts_A_pred - verts_B_pred) * (verts_A_pred - verts_B_pred)), axis=2)
    edgelenBC_pred = tf.reduce_sum(tf.sqrt((verts_C_pred - verts_B_pred) * (verts_C_pred - verts_B_pred)), axis=2)
    edgelenAC_pred = tf.reduce_sum(tf.sqrt((verts_A_pred - verts_C_pred) * (verts_A_pred - verts_C_pred)), axis=2)

    print ('edgelenAB', edgelenAB.shape, trismask.shape)

    if use_l2:
        dist = tf.boolean_mask(((tf.log(edgelenAB) - tf.log(edgelenAB_pred))**2 + \
                               (tf.log(edgelenBC) - tf.log(edgelenBC_pred))**2 + \
                               (tf.log(edgelenAC) - tf.log(edgelenAC_pred))**2),
                               trismask)
        dist = tf.maximum(dist - margin**2, 0.)
    else:
        # dist = tf.abs(tf.log(e1) - self.log_e0)
        dist = tf.boolean_mask((tf.abs(tf.log(edgelenAB) - tf.log(edgelenAB_pred)) + \
                                tf.abs(tf.log(edgelenBC) - tf.log(edgelenBC_pred)) + \
                                tf.abs(tf.log(edgelenAC) - tf.log(edgelenAC_pred))),
                                trismask)
        dist = tf.maximum(dist - margin, 0.)
    loss = tf.reduce_mean(dist)
    end_points['edgelenAB_pred'] = loss 

    return loss, end_points

def get_edgedir_loss(src_mesh, pred_verts, end_points={}, use_l2=False, margin=np.log(2.)):
    ##Encourages dihedral angle to be 180 degrees.

    verts = src_mesh['verts']
    nverts = src_mesh['nverts']
    tris = src_mesh['tris'] # B x maxntris x 3
    ntris = src_mesh['ntris']
    vertsmask = src_mesh['vertsmask']
    trismask = tf.cast(tf.squeeze(src_mesh['trismask']), tf.bool)

    maxntris = src_mesh['tris'].get_shape()[1]
    batch_size = src_mesh['tris'].get_shape()[0]

    B = np.zeros([batch_size, maxntris, 1], dtype=np.int64)
    for iB in range(batch_size):
        B[iB,:,:] = iB
    B = tf.constant(B)
    tris = tf.cast(tris, tf.int64)

    # print('tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 0], axis=-1)])', tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 0], axis=-1)]).shape)

    verts_A = tf.boolean_mask(tf.gather_nd(verts, tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 0], axis=-1)])))
    verts_B = tf.boolean_mask(tf.gather_nd(verts, tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 1], axis=-1)])))
    verts_C = tf.boolean_mask(tf.gather_nd(verts, tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 2], axis=-1)])))

    edgelenAB = tf.reduce_sum(tf.sqrt((verts_A - verts_B) * (verts_A - verts_B)), axis=2)
    edgelenBC = tf.reduce_sum(tf.sqrt((verts_C - verts_B) * (verts_C - verts_B)), axis=2)
    edgelenAC = tf.reduce_sum(tf.sqrt((verts_A - verts_C) * (verts_A - verts_C)), axis=2)

    verts_A_pred = tf.gather_nd(pred_verts, tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 0], axis=-1)]))
    verts_B_pred = tf.gather_nd(pred_verts, tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 1], axis=-1)]))
    verts_C_pred = tf.gather_nd(pred_verts, tf.concat(axis=2, values=[B, tf.expand_dims(tris[:, :, 2], axis=-1)]))

    edgelenAB_pred = tf.reduce_sum(tf.sqrt((verts_A_pred - verts_B_pred) * (verts_A_pred - verts_B_pred)), axis=2)
    edgelenBC_pred = tf.reduce_sum(tf.sqrt((verts_C_pred - verts_B_pred) * (verts_C_pred - verts_B_pred)), axis=2)
    edgelenAC_pred = tf.reduce_sum(tf.sqrt((verts_A_pred - verts_C_pred) * (verts_A_pred - verts_C_pred)), axis=2)

    non_zero_mask_AB =  tf.logical_and(tf.not_equal(edgelenAB, 0.), tf.not_equal(edgelenAB_pred, 0.))
    non_zero_mask_BC =  tf.logical_and(tf.not_equal(edgelenBC, 0.), tf.not_equal(edgelenBC_pred, 0.))
    non_zero_mask_AC =  tf.logical_and(tf.not_equal(edgelenAC, 0.), tf.not_equal(edgelenAC_pred, 0.))
    # print ('edgelenAB', edgelenAB.shape, trismask.shape)

    if use_l2:
        dist = (tf.boolean_mask((tf.log(edgelenAB) - tf.log(edgelenAB_pred))**2, non_zero_mask_AB) + \
                tf.boolean_mask((tf.log(edgelenBC) - tf.log(edgelenBC_pred))**2, non_zero_mask_BC) + \
                tf.boolean_mask((tf.log(edgelenAC) - tf.log(edgelenAC_pred))**2, non_zero_mask_AC))        
        dist = tf.maximum(dist - margin**2, 0.)
    else:
        # dist = tf.abs(tf.log(e1) - self.log_e0)
        dist = (tf.boolean_mask(tf.abs(tf.log(edgelenAB) - tf.log(edgelenAB_pred)), non_zero_mask_AB) + \
                tf.boolean_mask(tf.abs(tf.log(edgelenBC) - tf.log(edgelenBC_pred)), non_zero_mask_BC) + \
                tf.boolean_mask(tf.abs(tf.log(edgelenAC) - tf.log(edgelenAC_pred)), non_zero_mask_AC))
        dist = tf.maximum(dist - margin, 0.)
    loss = tf.reduce_mean(dist)
    end_points['edgelenAB_pred'] = loss 

    return loss, end_points

def get_render_loss(src_mesh, pred_verts, ref_mask, end_points={}):
    import neural_renderer

    src_nverts = src_mesh['nverts']
    src_tris = src_mesh['tris'] # B x maxntris x 3
    src_ntris = src_mesh['ntris']

    with tf.variable_scope("neuralrenderer") as scope:  
        renderer = neural_renderer.Renderer()

        pred_nmr = tf_neuralrenderer.NMR(renderer)
        pred_verts = tf_neuralrenderer.orthographic_proj_withz(pred_verts, cam)
        pred_renderedmask = pred_nmr.neural_renderer_mask(pred_verts, src_nverts, src_tris, src_ntris, 'pred_render_mask')
        pred_renderedmask = tf.expand_dims(pred_renderedmask, axis = 3)

    mask_loss = tf.reduce_mean(tf.nn.l2_loss(ref_mask - pred_renderedmask))

    return mask_loss, end_points

def get_render_loss_multiangle(src_mesh, pred_verts, ref_mesh, end_points={}, opt = 'img'): # multiview rendering loss
    import neural_renderer

    src_nverts = src_mesh['nverts']
    src_tris = src_mesh['tris'] # B x maxntris x 3
    src_ntris = src_mesh['ntris']

    ref_verts = ref_mesh['verts']
    ref_nverts = ref_mesh['nverts']
    ref_tris = ref_mesh['tris']
    ref_ntris = ref_mesh['ntris']

    with tf.variable_scope("neuralrenderer") as scope:     
        renderer = neural_renderer.Renderer()
        loss = 0

        for ir, r in enumerate([0, 1./5, 2./5, 3./5, 4./5]):    
            with tf.variable_scope('%s' % ir) as scope:   
                ref_nmr = tf_neuralrenderer.NMR(renderer)
                pred_nmr = tf_neuralrenderer.NMR(renderer)
                q = Quaternion(axis=[0, 1, 0], angle=r*2*np.pi)
                cams0 = np.array([[1.000000, 1.184425, 0.383219, -1.292325]], dtype=np.float32)#np.array([[1., 0.0064617, 0.04346868, -0.28036675]], dtype=np.float32)
                cams0 = tf.constant(cams0)
                cams1 = np.array([[q[0],q[1],q[2],q[3]]], dtype=np.float32)
                cams1 = tf.constant(cams1)
                cam = tf.concat([cams0, cams1], axis=1)

                ref_verts_ = tf_neuralrenderer.orthographic_proj_withz(ref_verts, cam)
                if opt == 'mask':
                    ref_renderedmask = ref_nmr.neural_renderer_mask(ref_verts_, ref_nverts, ref_tris, ref_ntris, 'ref_render_mask_%s' % ir)#output: BxHxW 
                    ref_renderedmask = tf.expand_dims(ref_renderedmask, axis = 3)  

                    pred_verts_ = tf_neuralrenderer.orthographic_proj_withz(pred_verts, cam)
                    pred_renderedmask = pred_nmr.neural_renderer_mask(pred_verts_, src_nverts, src_tris, src_ntris, 'pred_render_mask_%s' % ir)
                    pred_renderedmask = tf.expand_dims(pred_renderedmask, axis = 3)
                    loss += tf.reduce_mean(tf.nn.l2_loss(ref_renderedmask - pred_renderedmask))
                else:
                    ref_renderedmask = ref_nmr.neural_renderer_texture(ref_verts_, ref_nverts, ref_tris, ref_ntris, None, 'ref_render_mask_%s' % ir)#output: BxHxW 
                    print ('ref_renderedmask', ref_renderedmask.shape)
                    # ref_renderedmask = tf.expand_dims(ref_renderedmask)  
                    pred_verts_ = tf_neuralrenderer.orthographic_proj_withz(pred_verts, cam)
                    pred_renderedmask = pred_nmr.neural_renderer_texture(pred_verts_, src_nverts, src_tris, src_ntris, None, 'pred_render_mask_%s' % ir)
                    # pred_renderedmask = tf.expand_dims(pred_renderedmask)
                    loss += tf.reduce_mean(tf.nn.l2_loss(ref_renderedmask - pred_renderedmask))

    end_points['ref_renderedmask'] = ref_renderedmask     
    end_points['pred_renderedmask'] = pred_renderedmask     

    return loss, end_points


def get_render_loss(src_mesh, pred_verts, ref_mesh, end_points={}, opt = 'img'): # multiview rendering loss

    src_nverts = src_mesh['nverts']
    src_tris = src_mesh['tris'] # B x maxntris x 3
    src_ntris = src_mesh['ntris']

    ref_verts = ref_mesh['verts']
    ref_nverts = ref_mesh['nverts']
    ref_tris = ref_mesh['tris']
    ref_ntris = ref_mesh['ntris']
    cam = ref_mesh['cams']

    with tf.variable_scope("neuralrenderer") as scope:     
        renderer = neural_renderer.Renderer()
        loss = 0

        ref_nmr = tf_neuralrenderer.NMR(renderer)
        pred_nmr = tf_neuralrenderer.NMR(renderer)

        ref_verts_ = tf_neuralrenderer.orthographic_proj_withz(ref_verts, cam)

        if opt == 'mask':
            ref_renderedmask = ref_nmr.neural_renderer_mask(ref_verts_, ref_nverts, ref_tris, ref_ntris, 'ref_render_mask')#output: BxHxW 
            ref_renderedmask = tf.expand_dims(ref_renderedmask, axis = 3)  

            pred_verts_ = tf_neuralrenderer.orthographic_proj_withz(pred_verts, cam)
            pred_renderedmask = pred_nmr.neural_renderer_mask(pred_verts_, src_nverts, src_tris, src_ntris, 'pred_render_mask')
            pred_renderedmask = tf.expand_dims(pred_renderedmask, axis = 3)
            loss = tf.reduce_mean(tf.nn.l2_loss(ref_renderedmask - pred_renderedmask))
            
            end_points['ref_rendered'] = ref_renderedimg     
            end_points['pred_rendered'] = pred_renderedmask    

        else:
            ref_renderedimg = ref_nmr.neural_renderer_texture(ref_verts_, ref_nverts, ref_tris, ref_ntris, None, 'ref_render_img')#output: BxHxW  
            pred_verts_ = tf_neuralrenderer.orthographic_proj_withz(pred_verts, cam)
            pred_renderedimg = pred_nmr.neural_renderer_texture(pred_verts_, src_nverts, src_tris, src_ntris, None, 'pred_render_img')

            loss = tf.reduce_mean(tf.nn.l2_loss(ref_renderedimg - pred_renderedimg))

            end_points['ref_rendered'] = ref_renderedimg     
            end_points['pred_rendered'] = pred_renderedimg     

    return loss, end_points
