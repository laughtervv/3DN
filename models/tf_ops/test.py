import numpy as np
import tensorflow as tf
import approxmatch.tf_approxmatch as tf_approxmatch
import nn_distance.tf_nndistance as tf_nndistance
import mesh_sampling.tf_meshsampling as tf_meshsampling
import mesh_laplacian.tf_meshlaplacian as tf_meshlaplacian
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR))
import tf_util

if __name__=='__main__':
    import numpy as np
    import random
    import time
    from tensorflow.python.ops.gradient_checker import compute_gradient
    random.seed(100)
    np.random.seed(100)


    def write_off(fn, verts, faces):
        file = open(fn, 'w')
        file.write('OFF\n')
        file.write('%d %d %d\n' % (len(verts), len(faces), 0))
        for vert in verts:
            file.write('%f %f %f\n' % (vert[0], vert[1], vert[2]))
            # verts.append([float(s) for s in readline().strip().split(' ')])
        for face in faces:
            file.write('3 %f %f %f\n' % (face[0], face[1], face[2]))
            # faces.append([int(s) for s in readline().strip().split(' ')][1:])
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
        # xyz1=np.random.randn(32,16384,3).astype('float32')
        # xyz2=np.random.randn(32,1024,3).astype('float32')
        src_vertices, src_faces = read_off('1bf710535121b17cf453cc5da9731a22.off')
        dst_vertices, dst_faces = read_off('/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/part_mesh/Models/Chair/1bcec47c5dc259ea95ca4adb70946a21.off')


        np.random.seed(int(time.time()))
        # r1 = tf.constant(np.random.random_sample((1, 40000)),dtype=tf.float32)
        # r2 = tf.constant(np.random.random_sample((1, 40000)),dtype=tf.float32)
        # r = tf.constant(np.random.random_sample((1, 40000)),dtype=tf.float32)
        r1 = tf.random_uniform([1, 40000],dtype=tf.float32)
        r2 = tf.random_uniform([1, 40000],dtype=tf.float32)
        r = tf.random_uniform([1, 40000],dtype=tf.float32)


        ##target
        dst_verts=tf.expand_dims(tf.constant(dst_vertices),0)
        dst_tris=tf.expand_dims(tf.constant(dst_faces),0)
        dst_feats=tf.expand_dims(tf.constant(dst_vertices),0)
        dst_nverts = tf.constant([[len(dst_vertices)]],dtype=tf.int32)
        dst_ntris = tf.constant([[len(dst_faces)]],dtype=tf.int32)
        dst_points, outfeats, correspondingfaces = tf_meshsampling.mesh_sampling(dst_verts, dst_nverts, dst_tris, dst_ntris, dst_feats, r, r1, r2)
        dst_points_val, dst_verts_val = sess.run([dst_points, dst_verts])
        np.savetxt('dst_points.xyz', np.squeeze(dst_points_val))

        # dst_points_val = np.concatenate((dst_points_val, dst_verts_val), axis =1)
        targetpoints = tf.constant(dst_points_val)

        ##source
        src_verts=tf.expand_dims(tf.constant(src_vertices),0)
        src_tris=tf.expand_dims(tf.constant(src_faces),0)
        src_feats=tf.expand_dims(tf.constant(src_vertices),0)
        src_nverts = tf.constant([[len(src_vertices)]],dtype=tf.int32)
        src_ntris = tf.constant([[len(src_faces)]],dtype=tf.int32)
        src_points, outfeats, correspondingfaces = tf_meshsampling.mesh_sampling(src_verts, src_nverts, src_tris, src_ntris, src_feats, r, r1, r2)
        src_points_val, src_verts_val = sess.run([src_points, src_verts])
        np.savetxt('src_points.xyz', np.squeeze(src_points_val))

        # src_feats = tf.concat([src_feats]*100, axis=2)
        feats = tf.Variable(src_feats)
        print src_feats.get_shape()
        points, outfeats, correspondingfaces = tf_meshsampling.mesh_sampling(src_verts, src_nverts, src_tris, src_ntris, feats, r, r1, r2)
        laplacian1, _, _ = tf_meshlaplacian.mesh_laplacian(src_verts, src_nverts, src_tris, src_ntris)
        # feats1 = tf.slice(feats, [0, 0, 0], [1, -1, 3])
        laplacian2, _, _ = tf_meshlaplacian.mesh_laplacian(feats, src_nverts, src_tris, src_ntris)
        laplacian_loss = 10*tf.nn.l2_loss(laplacian1 - laplacian2)
        # outfeats = tf.concat([outfeats, feats], axis=1)

        #loss

        # outfeats = tf.expand_dims(outfeats, 2)

        # Encoder
        # net = tf_util.conv2d(outfeats, 3, [1,6],
        #                      padding='VALID', stride=[1,1],
        #                      bn=False,
        #                      scope='conv1')
        # net = tf.squeeze(outfeats, 2)
        # print net.get_shape()
        outfeats = tf.slice(outfeats, [0, 0, 0], [1, -1, 3])
        dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(outfeats, targetpoints)
        loss = 10000*tf.reduce_mean(dists_forward) + 10000*tf.reduce_mean(dists_backward)+laplacian_loss

        feats_grad = tf.gradients(loss, [feats])[0]
        #
        train=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

        sess.run(tf.initialize_all_variables())
        #
        old_lossval = 10000
        for  i in range(100):
            # feed_dict = {feats_ph: feats_val}
            _, loss_val, feats_val, points_val, dst_points_val, outfeats_val, feats_grad_val,correspondingfaces_val2,r_val =sess.run([train, loss, feats, points, targetpoints, outfeats, feats_grad, correspondingfaces,r])#, feed_dict=feed_dict)
            # print feats_val
            # feats_val -= feats_grad_val*0.005
            # if loss_val<old_lossval:
            #     old_lossval = loss_val
            # else:
            #     break
            print r_val
            print loss_val, np.argmax(feats_val[:,:,1]), np.min(feats_val[:,:,1]), np.max(dst_points_val[:,:,1]), np.min(dst_points_val[:,:,1])
            # print feats_grad_val[:,np.argmax(feats_val[:,:,1]),1]
            # print newpc_val[:,:,1]
            # print outfeats_val[:,:,1]
            # print feats_val.shape
            # print feats_val
            # print np.max(feats_grad_val[:,:,1]),np.min(feats_grad_val[:,:,1])
        #
        # np.savetxt('pts.xyz', points_val)
        np.savetxt('feats.xyz', np.squeeze(feats_val))
        np.savetxt('outfeats.xyz', np.squeeze(outfeats_val))
        write_off('out.off', np.squeeze(feats_val), src_faces)