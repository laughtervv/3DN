import argparse
import math
from datetime import datetime
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

slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='GPU to use [default: GPU 0]')
parser.add_argument('--category', default=None, help='Which single class to train on [default: None]')
parser.add_argument('--log_dir', default='checkpoint/3DN_allcategories_from2D', help='Log dir [default: log]')
parser.add_argument('--maxnverts', type=int, default=30000, help='Max number of vertices [default: 30000]')
parser.add_argument('--maxntris', type=int, default=100000, help='Max number of triangles [default: 70000]')
parser.add_argument('--num_points', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--restore_model', default='checkpoint/fromscratch', help='restore_model')
parser.add_argument('--restore_model3d', default='../3D/checkpoint/3DN_allcategories_from3D', help='restore_model')
parser.add_argument('--restore_modelcnn', default='../../models/CNN/pretrained_model/vgg_16.ckpt', help='restore_model')
parser.add_argument('--train_lst_mesh', default='/media/hdd2/data/ShapeNet/filelists/ShapeNetRendering/all_Rendering_template_30000_70000.lst', help='train mesh data list')
parser.add_argument('--valid_lst_mesh', default='/media/hdd2/data/ShapeNet/filelists/ShapeNetRendering/all_Rendering_template_30000_70000_valid.lst', help='test mesh data list')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

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
PRETRAINED_CNN_MODEL_FILE = FLAGS.restore_modelcnn
TRAIN_LST_MESH = FLAGS.train_lst_mesh
VALID_LST_MESH = FLAGS.valid_lst_mesh
LOG_DIR = FLAGS.log_dir
PRETRAINED_DEFORM3D_PATH = FLAGS.restore_model3d

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

RESULT_PATH = os.path.join(LOG_DIR, 'train_results')
if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)

VALID_RESULT_PATH = os.path.join(LOG_DIR, 'valid_results')
if not os.path.exists(VALID_RESULT_PATH): os.mkdir(VALID_RESULT_PATH)

os.system('cp %s.py %s' % (os.path.splitext(model.__file__)[0], LOG_DIR))
os.system('cp train_vgg.py %s' % (LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
IMG_SIZE = 137

HOSTNAME = socket.gethostname()

TRAIN_DATASET = data_rendered_img_template.RenderedImgDataset(TRAIN_LST_MESH, batch_size=BATCH_SIZE, img_size=(IMG_SIZE,IMG_SIZE), 
                                                              maxnverts=MAX_NVERTS, maxntris=MAX_NTRIS, normalize = True, 
                                                              h5folder='/media/hdd2/data/ShapeNet/ShapeNetCore.v2.h5')
VALID_DATASET = data_rendered_img_template.RenderedImgDataset(VALID_LST_MESH, batch_size=BATCH_SIZE, img_size=(IMG_SIZE,IMG_SIZE), 
                                                              maxnverts=MAX_NVERTS, maxntris=MAX_NTRIS, normalize = True, 
                                                              h5folder='/media/hdd2/data/ShapeNet/ShapeNetCore.v2.h5')

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
    learning_rate = tf.maximum(learning_rate, 1e-6, name='lr') # CLIP THE LEARNING RATE!
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

class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()

def load_model(sess, LOAD_MODEL_FILE, prefixs, strict=False):

    vars_in_pretrained_model = dict(checkpoint_utils.list_variables(LOAD_MODEL_FILE))
    vars_in_defined_model = []

    for var in tf.trainable_variables():
        if isinstance(prefixs, list):
            for prefix in prefixs:
                if (var.op.name.startswith(prefix)) and (var.op.name in vars_in_pretrained_model.keys()) and ('logits' not in var.op.name):
                    if (list(var.shape) == vars_in_pretrained_model[var.op.name]):
                        vars_in_defined_model.append(var)
        else:     
            if (var.op.name.startswith(prefixs)) and (var.op.name in vars_in_pretrained_model.keys()) and ('logits' not in var.op.name):
                if (list(var.shape) == vars_in_pretrained_model[var.op.name]):
                    vars_in_defined_model.append(var)

    saver = tf.train.Saver(vars_in_defined_model)
    try:
        saver.restore(sess, LOAD_MODEL_FILE)
        print( "Model loaded in file: %s" % (LOAD_MODEL_FILE))
    except:
        if strict:
            print( "Fail to load modelfile: %s" % LOAD_MODEL_FILE)
            return False
        else:
            print( "Model loaded in file: %s" % (LOAD_MODEL_FILE))
            return True

    return True

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            src_mesh = model.mesh_placeholder_inputs(BATCH_SIZE, MAX_NVERTS, MAX_NTRIS, img_size=(IMG_SIZE,IMG_SIZE), scope='src')
            ref_mesh = model.mesh_placeholder_inputs(BATCH_SIZE, MAX_NVERTS, MAX_NTRIS, img_size=(IMG_SIZE,IMG_SIZE), scope='ref')

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0,name='batch')
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss

            end_points = model.get_model(src_mesh, ref_mesh, NUM_POINTS, is_training_pl, bn=False)
            loss, end_points = model.get_loss(end_points)
            tf.summary.scalar('loss', loss)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            # Create a session
            config = tf.ConfigProto()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
            config=tf.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

            # Add summary writers
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

            ##### all
            update_variables = [x for x in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)]

            train_op = optimizer.minimize(loss, global_step=batch, var_list=update_variables)

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            ######### Loading Checkpoint ###############
            # CNN(Pretrained from ImageNet)
            if not load_model(sess, PRETRAINED_CNN_MODEL_FILE, 'vgg_16', strict=True):
                return

            # load weights from 3D deform net
            ckptstate = tf.train.get_checkpoint_state(PRETRAINED_DEFORM3D_PATH)
            if ckptstate is not None:
                PRETRAINED_DEFORM3D_MODEL_FILE = os.path.join(PRETRAINED_DEFORM3D_PATH, os.path.basename(ckptstate.model_checkpoint_path))
                load_model(sess, PRETRAINED_DEFORM3D_MODEL_FILE, ['sharebiasnet', 'srcpc', 'refpc'])

            # Overall  
            saver = tf.train.Saver([v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if('lr' not in v.name) and ('batch' not in v.name)])  
            ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)

            if ckptstate is not None:
                LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
                vars_in_pretrained_model = dict(checkpoint_utils.list_variables(LOAD_MODEL_FILE))
                checkpoint_keys = set(vars_in_pretrained_model.keys())
                current_keys = set([v.name.encode('ascii','ignore') for v in tf.global_variables()])
                try:
                    with NoStdStreams():
                        saver.restore(sess, LOAD_MODEL_FILE)
                    print( "Model loaded in file: %s" % LOAD_MODEL_FILE)    
                except:
                    print( "Fail to load overall modelfile: %s" % PRETRAINED_MODEL_PATH)

            ###########################################

            ops = {'src_mesh': src_mesh,
                   'ref_mesh': ref_mesh,
                   'is_training_pl': is_training_pl,
                   'loss': loss,
                   'train_op': train_op,
                   'merged': merged,
                   'step': batch,
                   'end_points': end_points}

            best_loss = 1e20
            for epoch in range(MAX_EPOCH):
                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

                epoch_loss = train_one_epoch(sess, ops, train_writer, saver)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                    log_string("Model saved in file: %s" % save_path)

                # Save the variables to disk.
                if epoch % 10 == 0:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                    log_string("Model saved in file: %s" % save_path)

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

def train_one_epoch(sess, ops, train_writer, saver):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train samples
    num_batches = int(len(TRAIN_DATASET) / BATCH_SIZE)

    random.shuffle(TRAIN_DATASET.order)
    TRAIN_DATASET.order = TRAIN_DATASET.order

    print('num_batches', num_batches)

    log_string(str(datetime.now()))

    loss_sum = 0
    loss_all = 0
    losses = {}
    for lossname in ops['end_points']['losses'].keys():
        losses[lossname] = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE

        src_batch_data, ref_batch_data = TRAIN_DATASET.get_batch(start_idx)

        feed_dict = {ops['is_training_pl']: is_training,}

        feed_src_mesh(ops['src_mesh'], src_batch_data, feed_dict)
        feed_ref_mesh(ops['ref_mesh'], ref_batch_data, feed_dict)

        output_list = [ops['train_op'], ops['merged'], ops['step'], ops['loss'], 
                       ops['end_points']['ref_pc'], ops['end_points']['pred_pc'], ops['end_points']['ref_img']]

        loss_list = []
        for il, lossname in enumerate(losses.keys()):
            loss_list += [ops['end_points']['losses'][lossname]]

        outputs = sess.run(output_list + loss_list, feed_dict=feed_dict)

        _, summary, step, loss_val, \
        ref_pc_val, pred_pc_val, ref_img_val = outputs[:-len(losses)]

        train_writer.add_summary(summary, step)

        for il, lossname in enumerate(losses.keys()):
            losses[lossname] += outputs[len(output_list)+il]

        loss_all += losses['overall_loss']

        save_freq = 1000
        if (batch_idx + 1) % save_freq == 0:
            save_path = saver.save(sess, os.path.join(LOG_DIR, "latest.ckpt"))

        verbose_freq = 500
        if (batch_idx + 1) % verbose_freq == 0:
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
            outstr = ' -- %03d / %03d -- ' % (batch_idx+1, num_batches)
            for lossname in losses.keys():
                outstr += '%s: %f, ' % (lossname, losses[lossname] / verbose_freq)
                losses[lossname] = 0
            log_string(outstr)


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    
    is_training = False

    # Shuffle train samples
    num_batches = int(len(VALID_DATASET)/BATCH_SIZE)

    print('num_batches', num_batches)
    print('len(TRAIN_DATASET_MESH)', len(VALID_DATASET))

    feed_dict = {ops['is_training_pl']: is_training,}
    loss_all = 0

    for batch_idx in range(num_batches):

        start_idx = batch_idx * BATCH_SIZE
        src_batch_data, ref_batch_data = VALID_DATASET.get_batch(start_idx)

        feed_dict = {ops['is_training_pl']: is_training,}
        feed_src_mesh(ops['src_mesh'], src_batch_data, feed_dict)
        feed_ref_mesh(ops['ref_mesh'], ref_batch_data, feed_dict)

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

        np.savetxt(os.path.join(VALID_RESULT_PATH, '%d_pc_gt.xyz' % batch_idx), ref_pc_val[bid,:,:])
        np.savetxt(os.path.join(VALID_RESULT_PATH, '%d_pc_pred.xyz' % batch_idx), pred_pc_val[bid,:,:])
        cv2.imwrite(os.path.join(VALID_RESULT_PATH, '%d_ref_img_resized.png' % batch_idx), (ref_img_val[bid,:,:,:] * 255).astype(np.uint8))

        data_off.write_off(os.path.join(VALID_RESULT_PATH, '%d_refmesh.off' % batch_idx),
                       ref_batch_data['verts'][bid, :ref_batch_data['nverts'][bid, 0], :],
                       ref_batch_data['tris'][bid, :ref_batch_data['ntris'][bid, 0], :])
        data_off.write_off(os.path.join(VALID_RESULT_PATH, '%d_srcmesh.off' % batch_idx),
                       src_batch_data['verts'][bid, :src_batch_data['nverts'][bid, 0], :],
                       src_batch_data['tris'][bid, :src_batch_data['ntris'][bid, 0], :])

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
