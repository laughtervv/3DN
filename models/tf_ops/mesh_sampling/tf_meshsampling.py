""" 
Mesh Sampling Operator
Author: Weiyue Wang
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import time
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mesh_sampling_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_meshsampling_so.so'))
BASE_DIR = os.path.dirname(BASE_DIR)
print BASE_DIR
sys.path.append(BASE_DIR)

def mesh_sampling(verts,nverts,tris,ntris,feats, r, r1, r2):
    '''
        Computes the distance of nearest neighbors for a pair of point clouds
        input: verts: (batch_size,max#verts,3)  vertices coordinates
        input: nverts: (batch_size,1)  vertices numbers
        input: tris: (batch_size,max#faces,3)  Triangle vertice indexes
        input: ntris: (batch_size,1)  Triangle numbers
        input: feats: (batch_size,#points,c)  vertice-wise features ----> Require GRAD
        input: R:  (batch_size,n_samples)   random number to sample points
        input: R1: (batch_size,n_samples)   random number 1 to sample points
        input: R2:  (batch_size,n_samples)   random number 2 to sample points
        output: points: (batch_size,n_samples,3)   points sampled from mesh
        output: outfeats:  (batch_size,n_samples,c)   output features for sampled points
        output: correpondingface:  (batch_size,n_samples)   sample points corresponding face indexes
    '''
    return mesh_sampling_module.mesh_sampling(verts,nverts,tris,ntris,feats, r, r1, r2)

@ops.RegisterGradient('MeshSampling')
def _mesh_sampling_grad(op, grad_points, grad_outfeats, grad_correpondingface):
    '''
    .Input("verts: float32")
    .Input("tris: int32")
    .Input("r1: float32")
    .Input("r2: float32")
    .Input("correspondingface: int32")
    .Input("grad_outfeats: float32")
    .Output("grad_feats: float32");
    '''
    verts = op.inputs[0]
    tris = op.inputs[2]
    r1 = op.inputs[6]
    r2 = op.inputs[7]
    correspondingface=op.outputs[2]
    global grad_feat
    grad_feat = mesh_sampling_module.mesh_sampling_grad(verts,tris,r1,r2,correspondingface,grad_outfeats)#[4]

    return [None,None,None,None,grad_feat,None,None,None]


if __name__=='__main__':
    import numpy as np
    import random
    import time

    import nn_distance.tf_nndistance as tf_nndistance
    import mesh_laplacian.tf_meshlaplacian as tf_meshlaplacian
    from tensorflow.python.ops.gradient_checker import compute_gradient
    random.seed(100)
    np.random.seed(100)


    def write_off(fn, verts, faces):
        file = open(fn, 'w')
        file.write('OFF\n')
        file.write('%d %d %d\n' % (len(verts), len(faces), 0))
        for vert in verts:
            file.write('%f %f %f\n' % (vert[0], vert[1], vert[2]))
        for face in faces:
            file.write('3 %f %f %f\n' % (face[0], face[1], face[2]))
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


    test1 = False
    test2 = True

    with tf.Session('') as sess:
        fh5 = h5py.File('/media/hdd2/data/ShapeNet/ShapeNetCore.v2.h5/03001627/1a6f615e8b1b5ae4dbbc9440457e303e.h5')

        vertices, faces = fh5['verts'], fh5['tris']
        with tf.device('/gpu:0'):
            if test1:
                r1 = tf.random_uniform([1, 20000])
                r2 = tf.random_uniform([1, 20000])
                r = tf.random_uniform([1, 20000])

                verts=tf.expand_dims(tf.constant(vertices),0)
                tris=tf.expand_dims(tf.constant(faces),0)
                feats=tf.expand_dims(tf.constant(vertices),0)
                nverts = tf.constant([[len(vertices)]],dtype=tf.int32)
                ntris = tf.constant([[len(faces)]],dtype=tf.int32)

                points, outfeats, correspondingfaces = mesh_sampling(verts, nverts, tris, ntris, feats, r, r1, r2)

                points_val = sess.run([points])

                points_val = np.squeeze(points_val[0])
                np.savetxt('tmp.xyz', points_val)

            if test2:

                verts0=tf.expand_dims(tf.constant(vertices),0)
                verts = tf.concat((verts0, verts0), axis = 0)
                verts0 = tf.concat((verts0, verts0), axis = 0)
                tris=tf.expand_dims(tf.constant(faces),0)
                tris = tf.concat((tris, tris), axis = 0)
                feats=tf.expand_dims(tf.constant(vertices),0)
                feats = tf.concat((feats, feats), axis = 0)
                nverts = tf.constant([[len(vertices)]],dtype=tf.int32)
                nverts = tf.concat((nverts, nverts), axis = 0)
                ntris = tf.constant([[len(faces)]],dtype=tf.int32)
                ntris = tf.concat((ntris, ntris), axis = 0)

                print verts.shape, nverts.shape

                np.random.seed(int(time.time()))
                r1 = tf.constant(np.random.random_sample((2, 20000)),dtype=tf.float32)
                r2 = tf.constant(np.random.random_sample((2, 20000)),dtype=tf.float32)
                r = tf.constant(np.random.random_sample((2, 20000)),dtype=tf.float32)

                points, outfeats, correspondingfaces = mesh_sampling(verts, nverts, tris, ntris, verts, r, r1, r2)
                for  i in range(1):
                    points_val,correspondingfaces_val,feats_val,verts_val = sess.run([points,correspondingfaces,verts,verts])
                points_val = points_val#[0]
                np.savetxt('Original.xyz', np.squeeze(points_val[0,...]))

                points_val = points_val
                points_val[0,:,1] *= 2
                points_val[1,:,1] += 0.05
                np.savetxt('feats0.xyz', np.squeeze(points_val[0,...]))
                np.savetxt('feats1.xyz', np.squeeze(points_val[1,...]))
                newpc = tf.constant(points_val)

                verts = tf.Variable(verts*1.1)

                laplacian1, _, _ = tf_meshlaplacian.mesh_laplacian(verts0, nverts, tris, ntris)
                laplacian2, _, _ = tf_meshlaplacian.mesh_laplacian(verts, nverts, tris, ntris)
                laplacian_loss = (laplacian1 - laplacian2)
                laplacian_loss = 0.01*tf.reduce_mean(tf.nn.l2_loss(laplacian_loss))

                points, outfeats, correspondingfaces = mesh_sampling(verts, nverts, tris, ntris, verts, r, r1, r2)

                dists_predpc, _, dists_refpc, _ = tf_nndistance.nn_distance(outfeats, newpc)
                pcloss = 10000* (tf.reduce_mean(dists_predpc) + tf.reduce_mean(dists_refpc))
                loss = pcloss + laplacian_loss

                feats_grad0 = tf.gradients(laplacian_loss, [verts])[0]
                feats_grad1 = tf.gradients(pcloss, [verts])[0]

                train = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss)

                sess.run(tf.initialize_all_variables())

                old_lossval = 10000
                print "start optimization"
                for  i in range(1000):
                    laplacian_loss_val, _, loss_val, feats_val, points_val, newpc_val, outfeats_val, correspondingfaces_val2, feats_grad_val0, feats_grad_val1 =sess.run([laplacian_loss,train, loss, verts, points, newpc, outfeats, correspondingfaces, feats_grad0, feats_grad1])#, feed_dict=feed_dict)

                    print(loss_val, laplacian_loss_val)

                np.savetxt('outfeats0.xyz', np.squeeze(outfeats_val[0,...]))
                write_off('out0.off', np.squeeze(feats_val[0,...]), faces)
                write_off('in.off', np.squeeze(vertices), faces)

                np.savetxt('outfeats1.xyz', np.squeeze(outfeats_val[1,...]))
                write_off('out1.off', np.squeeze(feats_val[1,...]), faces)


