import argparse
import numpy as np
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(os.path.dirname(BASE_DIR), 'data'))
import data_off
import pymesh
import cv2
import model_vgg as model

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='/home/laughtervv/Documents/projects/pointnet_ffd/shapenet/rendered_img/checkpoint/all_vgg_template_lap_recon_lr4_ftfromchair_multigpu_cont_fixbatch_fixspace_cont', help='Log dir [default: log]')
FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
PRETRAINED_MODEL_PATH = FLAGS.log_dir
NUM_POINTS = 2048
IMG_SIZE = 137

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

def pc_normalize(pc):

    """ pc: NxC, return NxC """
    l = pc.shape[0]

    centroid = np.mean(pc, axis=0)

    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))

    pc = pc / m

    return pc

def demo():
    with tf.device('/gpu:0'):
        for i in range(2,4):
            src_verts, src_tris = data_off.read_off('%02d_input.off' % i)
            src_verts = pc_normalize(src_verts)
            mesh = pymesh.form_mesh(src_verts, src_tris)
            mesh, _ = pymesh.split_long_edges(mesh, 0.03)
            src_verts, src_tris = mesh.vertices, mesh.faces
            ref_img = cv2.imread('%02d_ref.png' % i, cv2.IMREAD_UNCHANGED)[:,:,:3].astype(np.float32) / 255.
            # ref_img = cv2.imresize(ref_img, (137,137))
            # cv2.imwrite('%02d_ref.png' % i, (ref_img * 255).astype(np.int8))
            # print(ref_img.shape)
            # print(ref_img.shape)

            src_mesh = model.mesh_placeholder_inputs(1, src_verts.shape[0], src_tris.shape[0], (IMG_SIZE,IMG_SIZE), 'src')
            ref_mesh = model.mesh_placeholder_inputs(1, src_verts.shape[0], src_tris.shape[0], (IMG_SIZE,IMG_SIZE), 'ref')
            is_training_pl = tf.placeholder(tf.bool, shape=())

            end_points = model.get_model(src_mesh, ref_mesh, NUM_POINTS, is_training_pl)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            saver = tf.train.Saver()  
            ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
            if ckptstate is not None:
                LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
                saver.restore(sess, LOAD_MODEL_FILE)
                print( "Model loaded in file: %s" % LOAD_MODEL_FILE)

            else:
                print( "Fail to load modelfile: %s" % PRETRAINED_MODEL_PATH)
                return


            feed_dict = {is_training_pl: False,}
            feed_dict[src_mesh['verts']] = np.expand_dims(src_verts, axis = 0)
            feed_dict[src_mesh['nverts']] = np.expand_dims([src_verts.shape[0]], axis = 0)
            feed_dict[src_mesh['tris']] = np.expand_dims(src_tris, axis = 0)
            feed_dict[src_mesh['ntris']] = np.expand_dims([src_tris.shape[0]], axis = 0)

            feed_dict[ref_mesh['verts']] = np.expand_dims(src_verts, axis = 0) # not using
            feed_dict[ref_mesh['nverts']] = np.expand_dims([src_verts.shape[0]], axis = 0) # not using
            feed_dict[ref_mesh['tris']] = np.expand_dims(src_tris, axis = 0) # not using
            feed_dict[ref_mesh['ntris']] = np.expand_dims([src_tris.shape[0]], axis = 0) # not using
            feed_dict[ref_mesh['imgs']] = np.expand_dims(ref_img, axis = 0)

            
            pred_verts_val = sess.run(end_points['pred_verts'], feed_dict=feed_dict)
            data_off.write_off('%02d_deformed.off' % i, pred_verts_val[0,:,:], src_tris)
            tf.reset_default_graph()


if __name__ == "__main__":
    demo()