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
import data_h5_template
import output_utils
import model as model

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='GPU to use [default: GPU 0]')
parser.add_argument('--category', default=None, help='Which single class to train on [default: None]')
parser.add_argument('--log_dir', default='checkpoint/3DN_allcategories_from3D', help='Log dir [default: log]')
parser.add_argument('--maxnverts', type=int, default=25000, help='Point Number [default: 2048]')
parser.add_argument('--maxntris', type=int, default=200000, help='Point Number [default: 2048]')
parser.add_argument('--num_points', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--restore_model', default='checkpoint/3DN_chair_from3D', help='restore_model')
parser.add_argument('--train_lst_mesh', default='/media/hdd2/data/ShapeNet/filelists/ShapeNetCore.v2.h5/all_template_train_50000_200000.lst', help='train mesh data list')
parser.add_argument('--valid_lst_mesh', default='/media/hdd2/data/ShapeNet/filelists/ShapeNetCore.v2.h5/all_template_valid_50000_200000.lst', help='test mesh data list')
parser.add_argument('--decay_step', type=int, default=100000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--rotation', action='store_true', help='Disable random rotation during training.')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

NUM_PART_CATEGORIES = 4

BATCH_SIZE = FLAGS.batch_size
MAX_NVERTS = FLAGS.maxnverts
MAX_NTRIS = FLAGS.maxntris
NUM_POINTS = FLAGS.num_points
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
PRETRAINED_MODEL_PATH = FLAGS.restore_model
TRAIN_LST_MESH = FLAGS.train_lst_mesh
VALID_LST_MESH = FLAGS.valid_lst_mesh


os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

RESULT_PATH = os.path.join(LOG_DIR, 'train_results')
if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)

VALID_RESULT_PATH = os.path.join(LOG_DIR, 'valid_results')
if not os.path.exists(VALID_RESULT_PATH): os.mkdir(VALID_RESULT_PATH)

os.system('cp %s.py %s' % (os.path.splitext(model.__file__)[0], LOG_DIR))
os.system('cp train.py %s' % (LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_%s.txt' % str(datetime.now()) ), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

TRAIN_DATASET_MESH = data_h5_template.H5Dataset(listfile=TRAIN_LST_MESH, maxnverts=MAX_NVERTS, maxntris=MAX_NTRIS, num_points=NUM_POINTS, batch_size=BATCH_SIZE)
VALID_DATASET_MESH = data_h5_template.H5Dataset(listfile=VALID_LST_MESH, maxnverts=MAX_NVERTS, maxntris=MAX_NTRIS, num_points=NUM_POINTS, batch_size=BATCH_SIZE)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001, name='lr') # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            src_mesh = model.mesh_placeholder_inputs(BATCH_SIZE, MAX_NVERTS, MAX_NTRIS, NUM_POINTS, 'src')
            ref_mesh = model.mesh_placeholder_inputs(BATCH_SIZE, MAX_NVERTS, MAX_NTRIS, NUM_POINTS, 'ref')
            refpc_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINTS, 3))
            vertsmask_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, MAX_NVERTS, 1))

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0, name='batch')
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss

            end_points = model.get_model(src_mesh, ref_mesh, NUM_POINTS, is_training_pl)
            loss, end_points = model.get_loss(end_points, NUM_PART_CATEGORIES)
            tf.summary.scalar('loss', loss)

            update_variables = [x for x in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)] 

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch, var_list=update_variables)

            # Add ops to save and restore all the variables.

            # Create a session
            # with tf.device('/gpu:0'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

            # Add summary writers
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            saver = tf.train.Saver([v for v in tf.all_variables() if('lr' not in v.name) and ('batch' not in v.name)])  
            ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
            if ckptstate is not None:
                try:
                    LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
                    saver.restore(sess, LOAD_MODEL_FILE)
                    print( "Model loaded in file: %s" % LOAD_MODEL_FILE)
                except:
                    check_var_list = tf.train.list_variables(LOAD_MODEL_FILE)
                    check_var_list = [x[0] for x in check_var_list]
                    check_var_set = set(check_var_list)
                    check_var_list = [x for x in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if ((x.name[:x.name.index(":")] in check_var_set) and ('batch' not in x.name))]
                    loader = tf.train.Saver(check_var_list)
                    loader.restore(sess, LOAD_MODEL_FILE)
                    print( "Model loaded in file: %s with missing variables" % LOAD_MODEL_FILE)

            else:
                print( "Fail to load modelfile: %s" % PRETRAINED_MODEL_PATH)
            # sess.run(init, {is_training_pl: True})

            ops = {'src_mesh': src_mesh,
                   'ref_mesh': ref_mesh,
                   'refpc_pl': refpc_pl,
                   'vertsmask_pl': vertsmask_pl,
                   'is_training_pl': is_training_pl,
                   'end_points': end_points,
                   'loss': loss,
                   'train_op': train_op,
                   'merged': merged,
                   'step': batch,
                   'end_points': end_points}

            best_loss = 1e20
            eval_one_epoch(sess, ops, test_writer)
            for epoch in range(MAX_EPOCH):
                log_string('**** EPOCH %03d ****' % (epoch + 1))
                sys.stdout.flush()

                train_one_epoch(sess, ops, train_writer)

                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

                epoch_loss = eval_one_epoch(sess, ops, test_writer)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                    log_string("Model saved in file: %s" % save_path)


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

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train samples
    random.shuffle(TRAIN_DATASET_MESH.order)
    # TRAIN_DATASET_MESH2.order = TRAIN_DATASET_MESH1.order
    num_batches = len(TRAIN_DATASET_MESH) // BATCH_SIZE

    # print('len(TRAIN_DATASET_MESH)', len(TRAIN_DATASET_MESH1))

    log_string(str(datetime.now()))

    loss_sum = 0
    losses = {}
    for lossname in ops['end_points']['losses'].keys():
        losses[lossname] = 0
    loss_all = 0
    tic = time.time()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE

        feed_dict = {ops['is_training_pl']: is_training,}

        src_batch_data, ref_batch_data = TRAIN_DATASET_MESH.get_batch(start_idx)

        feed_mesh(ops['src_mesh'], src_batch_data, feed_dict)
        feed_mesh(ops['ref_mesh'], ref_batch_data, feed_dict)

        output_list = [ops['train_op'], ops['merged'], ops['step'], ops['end_points']['ref_pc'], ops['end_points']['src_pc'], ops['end_points']['pred_pc'], ops['end_points']['pred_verts']]

        loss_list = []
        for il, lossname in enumerate(losses.keys()):
            loss_list += [ops['end_points']['losses'][lossname]]

        outputs = sess.run(output_list + loss_list, feed_dict=feed_dict)

        _, summary, step, ref_pc_val, src_pc_val, pred_pc_val, pred_verts_val = outputs[:-len(losses)]

        train_writer.add_summary(summary, step)
        loss_all += losses['overall_loss']

        for il, lossname in enumerate(losses.keys()):
            losses[lossname] += outputs[len(output_list)+il]

        verbose_freq = 50

        if (batch_idx+1) % verbose_freq == 0:
            bid = np.random.randint(BATCH_SIZE)
            data_off.write_off(os.path.join(RESULT_PATH, '%d_inputmesh.off' % batch_idx),
                           src_batch_data['verts'][bid, :src_batch_data['nverts'][bid, 0], :],
                           src_batch_data['tris'][bid, :src_batch_data['ntris'][bid, 0], :])
            data_off.write_off(os.path.join(RESULT_PATH, '%d_deformmesh.off' % batch_idx),
                           pred_verts_val[bid, :src_batch_data['nverts'][bid, 0], :],
                           src_batch_data['tris'][bid, :src_batch_data['ntris'][bid, 0], :])
            data_off.write_off(os.path.join(RESULT_PATH, '%d_refmesh.off' % batch_idx),
                           ref_batch_data['verts'][bid, :ref_batch_data['nverts'][bid, 0], :],
                           ref_batch_data['tris'][bid, :ref_batch_data['ntris'][bid, 0], :])
            np.savetxt(os.path.join(RESULT_PATH, '%d_ref.xyz' % batch_idx), ref_pc_val[bid,:,:])
            np.savetxt(os.path.join(RESULT_PATH, '%d_src.xyz' % batch_idx), src_pc_val[bid,:,:])
            np.savetxt(os.path.join(RESULT_PATH, '%d_pcpred.xyz' % batch_idx), pred_pc_val[bid,:,:])

            outstr = ' -- %03d / %03d -- ' % (batch_idx+1, num_batches)
            for lossname in losses.keys():
                outstr += '%s: %f, ' % (lossname, losses[lossname] / verbose_freq)
                losses[lossname] = 0

            outstr += 'time: %f, ' % (time.time() - tic)
            tic = time.time()
            log_string(outstr)

def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    
    is_training = False

    # Shuffle train samples
    num_batches = int(len(VALID_DATASET_MESH)/BATCH_SIZE)

    print('num_batches', num_batches)
    print('len(TRAIN_DATASET_MESH)', len(VALID_DATASET_MESH))

    feed_dict = {ops['is_training_pl']: is_training,}
    loss_all = 0

    for batch_idx in range(num_batches):

        start_idx = batch_idx * BATCH_SIZE
        src_batch_data, ref_batch_data = VALID_DATASET_MESH.get_batch(start_idx)

        if FLAGS.rotation:
            rotation_angle = np.random.uniform() * 2 * np.pi
            src_batch_data['verts'] = rotate_point_cloud(src_batch_data['verts'], rotation_angle)
            ref_batch_data['verts'] = rotate_point_cloud(ref_batch_data['verts'], rotation_angle)

        feed_dict = {ops['is_training_pl']: is_training,}
        feed_mesh(ops['src_mesh'], src_batch_data, feed_dict)
        feed_mesh(ops['ref_mesh'], ref_batch_data, feed_dict)

        loss_val, \
        ref_pc_val, \
        pred_verts_val, \
        pred_pc_val = sess.run(\
        [ops['end_points']['losses']['overall_loss'],
         ops['end_points']['ref_pc'],
         ops['end_points']['pred_verts'], 
         ops['end_points']['pred_pc']], feed_dict=feed_dict)

        loss_all += loss_val
        bid = 0
        data_off.write_off(os.path.join(VALID_RESULT_PATH, '%d_deformmesh.off' % batch_idx),
                       pred_verts_val[bid, :src_batch_data['nverts'][bid, 0], :],
                       src_batch_data['tris'][bid, :src_batch_data['ntris'][bid, 0], :])
        data_off.write_off(os.path.join(VALID_RESULT_PATH, '%d_inputmesh.off' % batch_idx),
                       src_batch_data['verts'][bid, :src_batch_data['nverts'][bid, 0], :],
                       src_batch_data['tris'][bid, :src_batch_data['ntris'][bid, 0], :])
        data_off.write_off(os.path.join(VALID_RESULT_PATH, '%d_refmesh.off' % batch_idx),
                       ref_batch_data['verts'][bid, :ref_batch_data['nverts'][bid, 0], :],
                       ref_batch_data['tris'][bid, :ref_batch_data['ntris'][bid, 0], :])
        np.savetxt(os.path.join(VALID_RESULT_PATH, '%d_pcpred.xyz' % batch_idx), pred_pc_val[bid,:,:])
    log_string("validation loss: %f" % (loss_all/num_batches))

    return loss_all/num_batches

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()