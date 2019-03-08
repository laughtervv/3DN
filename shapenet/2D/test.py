# CUDA_CACHE_MAXSIZE=$((1024*1024*1024)) python train_inputmesh.py to prevent tf freeze
import argparse
import math
from datetime import datetime
# import h5py
import numpy as np
import random
import tensorflow as tf
import socket
import importlib
import os
import cv2
import sys
import time
from tensorflow.contrib.framework.python.framework import checkpoint_utils
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(os.path.join(os.path.dirname(BASE_DIR), 'data'))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(os.path.dirname(BASE_DIR), 'data'))
print(os.path.join(ROOT_DIR, 'data'))
import data_rendered_img_template 
import model_vgg as model
import data_off
import data_h5
import test_utils


slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='2', help='GPU to use [default: GPU 0]')
parser.add_argument('--category', default=None, help='Which single class to train on [default: None]')
parser.add_argument('--log_dir', default='checkpoint/3DN_allcategories_from2D', help='Log dir [default: log]')
parser.add_argument('--maxnverts', type=int, default=35000, help='Max number of vertices [default: 30000]')
parser.add_argument('--maxntris', type=int, default=200000, help='Max number of triangles [default: 70000]')
parser.add_argument('--num_points', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--output_verbose', action='store_true', help='Save output files.')
parser.add_argument('--test_lst_mesh', default='/media/hdd2/data/ShapeNet/filelists/ShapeNetRendering/02691156_official_test.lst', help='test mesh data list')

FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
GPU_INDEX = FLAGS.gpu
MAX_NVERTS = FLAGS.maxnverts
MAX_NTRIS = FLAGS.maxntris
NUM_POINTS = FLAGS.num_points
PRETRAINED_MODEL_PATH = FLAGS.log_dir
TEST_LST_MESH = FLAGS.test_lst_mesh
LOG_DIR = FLAGS.log_dir

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

RESULT_PATH = os.path.join(LOG_DIR, 'test_results')
if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)

os.system('cp %s.py %s' % (os.path.splitext(model.__file__)[0], LOG_DIR))
os.system('cp train_vgg.py %s' % (LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

IMG_SIZE = 137

HOSTNAME = socket.gethostname()
TEST_DATASET = data_rendered_img_template.RenderedImgDataset(TEST_LST_MESH, batch_size=BATCH_SIZE, img_size=(IMG_SIZE,IMG_SIZE), 
                                                              maxnverts=MAX_NVERTS, maxntris=MAX_NTRIS, normalize = True, 
                                                              h5folder='/media/hdd2/data/ShapeNet/ShapeNetCore.v2.h5')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            src_mesh = model.mesh_placeholder_inputs(BATCH_SIZE, MAX_NVERTS, MAX_NTRIS, img_size=(IMG_SIZE,IMG_SIZE), scope='src')
            ref_mesh = model.mesh_placeholder_inputs(BATCH_SIZE, MAX_NVERTS, MAX_NTRIS, img_size=(IMG_SIZE,IMG_SIZE), scope='ref')

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            print("--- Get model and loss")
            # Get model and loss

            end_points = model.get_model(src_mesh, ref_mesh, NUM_POINTS, is_training_pl, bn=False)
            loss, end_points = model.get_loss(end_points)

            # Create a session
            config = tf.ConfigProto()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
            config=tf.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            ######### Loading Checkpoint ###############
            # Overall  
            saver = tf.train.Saver()  
            ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)

            if ckptstate is not None:
                LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
                saver.restore(sess, LOAD_MODEL_FILE)
                print( "Model loaded in file: %s" % LOAD_MODEL_FILE)    
            else:
                print( "Fail to load overall modelfile: %s" % PRETRAINED_MODEL_PATH)
                return

            ###########################################

            ops = {'src_mesh': src_mesh,
                   'ref_mesh': ref_mesh,
                   'is_training_pl': is_training_pl,
                   'end_points': end_points}

            test_(sess, ops)

def feed_src_mesh(mesh_ph, batch_data, feed_dict):
    feed_dict[mesh_ph['verts']] = batch_data['verts']
    feed_dict[mesh_ph['nverts']] = batch_data['nverts']
    feed_dict[mesh_ph['tris']] = batch_data['tris']
    feed_dict[mesh_ph['ntris']] = batch_data['ntris']

def feed_ref_mesh(mesh_ph, batch_data, feed_dict):
    feed_dict[mesh_ph['verts']] = batch_data['verts']
    feed_dict[mesh_ph['nverts']] = batch_data['nverts']
    feed_dict[mesh_ph['tris']] = batch_data['tris']
    feed_dict[mesh_ph['ntris']] = batch_data['ntris']
    feed_dict[mesh_ph['imgs']] = batch_data['imgs']


def test_(sess, ops):
    """ ops: dict mapping from string to tf ops """
    
    is_training = False

    # Shuffle train samples
    num_batches = int(len(TEST_DATASET)/BATCH_SIZE)

    print('num_batches', num_batches)
    print('len(TRAIN_DATASET_MESH)', len(TEST_DATASET))

    feed_dict = {ops['is_training_pl']: is_training,}
    pc_cf_loss_val_all = 0
    pc_em_loss_val_all = 0
    mesh_cf_loss_val_all = 0
    mesh_em_loss_val_all = 0
    iou_all = 0

    for batch_idx in range(num_batches):

        start_idx = batch_idx * BATCH_SIZE
        src_batch_data, ref_batch_data = TEST_DATASET.get_batch(start_idx)

        feed_dict = {ops['is_training_pl']: is_training,}
        feed_src_mesh(ops['src_mesh'], src_batch_data, feed_dict)
        feed_ref_mesh(ops['ref_mesh'], ref_batch_data, feed_dict)

        pc_cf_loss_val, pc_em_loss_val, \
        mesh_cf_loss_val, mesh_em_loss_val, \
        ref_pc_val, \
        pred_verts_val, \
        pred_pc_val = sess.run(\
        [ops['end_points']['losses']['pc_cf_loss'],ops['end_points']['losses']['pc_em_loss'],
         ops['end_points']['losses']['mesh_cf_loss'],ops['end_points']['losses']['mesh_em_loss'],
         ops['end_points']['ref_pc'],
         ops['end_points']['pred_verts'], 
         ops['end_points']['pred_pc']], feed_dict=feed_dict)

        pc_cf_loss_val = pc_cf_loss_val/10.
        bid = 0
        iou = test_utils.iou_pymesh((pred_verts_val[bid, :src_batch_data['nverts'][bid, 0], :],\
                                   src_batch_data['tris'][bid, :src_batch_data['ntris'][bid, 0]]), \
                                  (ref_batch_data['verts'][bid, :ref_batch_data['nverts'][bid, 0], :],\
                                   ref_batch_data['tris'][bid, :ref_batch_data['ntris'][bid, 0], :]))

        pc_cf_loss_val_all += pc_cf_loss_val
        pc_em_loss_val_all += pc_em_loss_val
        mesh_cf_loss_val_all += mesh_cf_loss_val
        mesh_em_loss_val_all += mesh_em_loss_val
        iou_all += iou

        print("processing[%d/%d] pc cf: %f, pc em: %f, mesh cf: %f, mesh em: %f, iou: %f" % 
             (batch_idx, num_batches, pc_cf_loss_val, pc_em_loss_val, mesh_cf_loss_val, mesh_em_loss_val, iou))

        if FLAGS.output_verbose:

            bid = 0

            np.savetxt(os.path.join(RESULT_PATH, '%d_pc_gt.xyz' % batch_idx), ref_pc_val[bid,:,:])
            np.savetxt(os.path.join(RESULT_PATH, '%d_pc_pred.xyz' % batch_idx), pred_pc_val[bid,:,:])
            cv2.imwrite(os.path.join(RESULT_PATH, '%d_ref_img_resized.png' % batch_idx), (ref_img_val[bid,:,:,:] * 255).astype(np.uint8))

            data_off.write_off(os.path.join(RESULT_PATH, '%d_refmesh.off' % batch_idx),
                           ref_batch_data['verts'][bid, :ref_batch_data['nverts'][bid, 0], :],
                           ref_batch_data['tris'][bid, :ref_batch_data['ntris'][bid, 0], :])
            data_off.write_off(os.path.join(RESULT_PATH, '%d_srcmesh.off' % batch_idx),
                           src_batch_data['verts'][bid, :src_batch_data['nverts'][bid, 0], :],
                           src_batch_data['tris'][bid, :src_batch_data['ntris'][bid, 0], :])

            data_off.write_off(os.path.join(RESULT_PATH, '%d_deformmesh.off' % batch_idx),
                           pred_verts_val[bid, :src_batch_data['nverts'][bid, 0], :],
                           src_batch_data['tris'][bid, :src_batch_data['ntris'][bid, 0], :])
            data_off.write_off(os.path.join(RESULT_PATH, '%d_inputmesh.off' % batch_idx),
                           src_batch_data['verts'][bid, :src_batch_data['nverts'][bid, 0], :],
                           src_batch_data['tris'][bid, :src_batch_data['ntris'][bid, 0], :])
            data_off.write_off(os.path.join(RESULT_PATH, '%d_refmesh.off' % batch_idx),
                           ref_batch_data['verts'][bid, :ref_batch_data['nverts'][bid, 0], :],
                           ref_batch_data['tris'][bid, :ref_batch_data['ntris'][bid, 0], :])
            np.savetxt(os.path.join(RESULT_PATH, '%d_pcpred.xyz' % batch_idx), pred_pc_val[bid,:,:])

    log_string("test pc cf loss: %f" % (pc_cf_loss_val_all/num_batches))
    log_string("test pc em loss: %f" % (pc_em_loss_val_all/num_batches))
    log_string("test mesh cf loss: %f" % (mesh_cf_loss_val_all/num_batches))
    log_string("test mesh em loss: %f" % (mesh_em_loss_val_all/num_batches))
    log_string("test iou: %f" % (iou_all/num_batches))

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    test()
    LOG_FOUT.close()
