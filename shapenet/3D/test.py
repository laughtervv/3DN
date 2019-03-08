import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import random
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(os.path.join(os.path.dirname(BASE_DIR), 'data'))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(os.path.dirname(BASE_DIR), 'data'))
print (os.path.join(ROOT_DIR, 'data'))
import data_off
import data_obj
import data_h5_template
import output_utils
import test_utils
import model as model

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='checkpoint/3DN_allcategories_from3D', help='Log dir [default: log]')
parser.add_argument('--maxnverts', type=int, default=35000, help='Point Number [default: 2048]')
parser.add_argument('--maxntris', type=int, default=200000, help='Point Number [default: 2048]')
parser.add_argument('--num_points', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--test_lst_mesh', default='/media/hdd2/data/ShapeNet/filelists/ShapeNetCore.v2.h5/official/03691459_official_test.lst', help='test mesh data list')
parser.add_argument('--rotation', action='store_true', help='Disable random rotation during training.')
parser.add_argument('--output_verbose', action='store_true', help='Save output files.')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

NUM_PART_CATEGORIES = 4

BATCH_SIZE = FLAGS.batch_size
MAX_NVERTS = FLAGS.maxnverts
MAX_NTRIS = FLAGS.maxntris
NUM_POINTS = FLAGS.num_points
GPU_INDEX = FLAGS.gpu
PRETRAINED_MODEL_PATH = FLAGS.log_dir
TEST_LST_MESH = FLAGS.test_lst_mesh

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

RESULT_PATH = os.path.join(LOG_DIR, 'test_results')
if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)

os.system('cp %s.py %s' % (os.path.splitext(model.__file__)[0], LOG_DIR))
os.system('cp train.py %s' % (LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_%s.txt' % str(datetime.now()) ), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

HOSTNAME = socket.gethostname()

TEST_DATASET_MESH = data_h5_template.H5Dataset(listfile=TEST_LST_MESH, maxnverts=MAX_NVERTS, maxntris=MAX_NTRIS, num_points=NUM_POINTS, batch_size=BATCH_SIZE)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            src_mesh = model.mesh_placeholder_inputs(BATCH_SIZE, MAX_NVERTS, MAX_NTRIS, NUM_POINTS, 'src')
            ref_mesh = model.mesh_placeholder_inputs(BATCH_SIZE, MAX_NVERTS, MAX_NTRIS, NUM_POINTS, 'ref')

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            print("--- Get model")
            end_points = model.get_model(src_mesh, ref_mesh, NUM_POINTS, is_training_pl)
            loss, end_points = model.get_loss(end_points, NUM_PART_CATEGORIES)
            # Add ops to save and restore all the variables.

            # Create a session
            # with tf.device('/gpu:0'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

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

            ops = {'src_mesh': src_mesh,
                   'ref_mesh': ref_mesh,
                   'is_training_pl': is_training_pl,
                   'end_points': end_points}

            test_(sess, ops)


def feed_mesh(mesh_ph, batch_data, feed_dict):
    feed_dict[mesh_ph['verts']] = batch_data['verts']
    feed_dict[mesh_ph['nverts']] = batch_data['nverts']
    feed_dict[mesh_ph['tris']] = batch_data['tris']
    feed_dict[mesh_ph['ntris']] = batch_data['ntris']

def rotate_point_cloud(batch_data, rotation_angle):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in np.arange(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def test_(sess, ops):
    """ ops: dict mapping from string to tf ops """
    
    is_training = False

    # Shuffle train samples
    num_batches = int(len(TEST_DATASET_MESH)/BATCH_SIZE)

    print('num_batches', num_batches)
    print('len(TRAIN_DATASET_MESH)', len(TEST_DATASET_MESH))

    feed_dict = {ops['is_training_pl']: is_training,}
    pc_cf_loss_val_all = 0
    pc_em_loss_val_all = 0
    mesh_cf_loss_val_all = 0
    mesh_em_loss_val_all = 0
    iou_all = 0

    for batch_idx in range(num_batches):

        start_idx = batch_idx * BATCH_SIZE
        src_batch_data, ref_batch_data = TEST_DATASET_MESH.get_batch(start_idx)

        if FLAGS.rotation:
            rotation_angle = np.random.uniform() * 2 * np.pi
            src_batch_data['verts'] = rotate_point_cloud(src_batch_data['verts'], rotation_angle)
            ref_batch_data['verts'] = rotate_point_cloud(ref_batch_data['verts'], rotation_angle)

        feed_dict = {ops['is_training_pl']: is_training,}
        feed_mesh(ops['src_mesh'], src_batch_data, feed_dict)
        feed_mesh(ops['ref_mesh'], ref_batch_data, feed_dict)

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
            data_obj.write_obj(os.path.join(RESULT_PATH, '%d_deformmesh.obj' % batch_idx),
                           pred_verts_val[bid, :src_batch_data['nverts'][bid, 0], :],
                           src_batch_data['tris'][bid, :src_batch_data['ntris'][bid, 0], :])
            data_obj.write_obj(os.path.join(RESULT_PATH, '%d_inputmesh.obj' % batch_idx),
                           src_batch_data['verts'][bid, :src_batch_data['nverts'][bid, 0], :],
                           src_batch_data['tris'][bid, :src_batch_data['ntris'][bid, 0], :])
            data_obj.write_obj(os.path.join(RESULT_PATH, '%d_refmesh.obj' % batch_idx),
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
