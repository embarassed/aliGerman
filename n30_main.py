import tensorflow as tf
import time
import os
import h5py
import numpy as np
import logging
import sys
from tensorflow.python.training import moving_averages
import pandas as pd
import math
##[N.B.]python=2.7
################################################# note ###################################################
#new_version-30:
# write the model inference and training like 'resnet'.
# 'batch_size=64' +
# 'no scaler' +
# 'seperate sen1, sen2' +
# '7cnn-4max_pool(filter_size:4*4,pool:2*2)(featuremap:64-128-256-512)-4residual-1avg_pool+final_concat_1024d(no bias)' +
# 'cnn-batch normal' +
# 'tf.contrib.layers.xavier_initializer()' +
# 'dynamic adjust learning_rate, lr(init)1e-3' +
# 'one dropout(0.7 const) before output_layer'
#   65.4%(val) 66.4%(test)
##########################################################################################################
flags = tf.app.flags
FLAGS = flags.FLAGS
# about data:
flags.DEFINE_string('base_dir', '/home/hadoop/aliGerman', 'father folder of all this')
flags.DEFINE_string('train_dir', 'training.h5', 'train')  # need to concat base_dir
flags.DEFINE_string('valid_dir', 'validation.h5', 'valid') # need to concat base_dir
flags.DEFINE_string('test_dir', 'round1_test_a_20181109.h5', 'test') # need to concat base_dir
flags.DEFINE_string('outdir_prefix', 'n30_model', 'prefix of outdir(model,log,result)')
flags.DEFINE_string('log_dir', None, 'save model and result_of_test') # need to concat base_dir
flags.DEFINE_integer('H', 32, 'input figure height')
flags.DEFINE_integer('W', 32, 'input figure weight')
flags.DEFINE_integer('sen1_Ch', 8, 'sen1 channels')
flags.DEFINE_integer('sen2_Ch', 10, 'sen2 channels')
flags.DEFINE_integer('label_class', 17, 'classes of label(one-hot)')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
# about model:
flags.DEFINE_float('regular_decay', 1e-6, 'l2 regularization decay.')
flags.DEFINE_float('regular_weight_decay', 1.0, 'l2 regularization  weight decay.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate. -1: select by hyperopt tuning.')
flags.DEFINE_float('min_learning_rate', 5e-6, 'Minimum learning rate')
flags.DEFINE_float('lr_decay', 0.98, 'Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', 200, 'Interval beteween each deacy.')
flags.DEFINE_integer('lr_decay_begin', 1000, 'deacy start after steps.')
flags.DEFINE_integer('epoch_size', -1, '=samples//batch_size')
flags.DEFINE_float('keep_prob', 0.7, 'keep prob of dropout_layer at before output_layer')
flags.DEFINE_integer('epochs', 100, 'Maximum number of epochs to train.')
flags.DEFINE_integer('patience', 20, 'Max number of epochs allowed for non-improving validation error before early stopping.')

def get_logger(log_dir):
    logger = logging.getLogger(log_dir)
    logger.setLevel(logging.DEBUG)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, 'info.log'))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger

class StandardScaler:
    """
    Standard the input
    """
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std
    def update(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def _generate_run_id(config):
    outdir_prefix = config.get('outdir_prefix')
    batch_size = config.get('batch_size')
    learning_rate = config.get('learning_rate')
    regular_decay = config.get('regular_decay')
    dropout = config.get('keep_prob')
    min_lr = config.get('min_learning_rate')
    lr_decay = config.get('lr_decay')
    lr_decay_steps = config.get('lr_decay_steps')
    lr_decay_begin = config.get('lr_decay_begin')
    run_id = '%s_l2_%g_lr_%g_minlr_%g_lrd_%g_steps_%d_lrdbeg_%d_bs_%d_drop_%g_%s' % (
        outdir_prefix, regular_decay, learning_rate, min_lr, lr_decay, lr_decay_steps, lr_decay_begin, batch_size, dropout, time.strftime('%m%d%H%M%S'))
    return run_id

def fetch_global_config():
    c = dict()
    c['base_dir'] = FLAGS.base_dir
    c['train_dir'] = os.path.join(c['base_dir'], FLAGS.train_dir)
    c['valid_dir'] = os.path.join(c['base_dir'], FLAGS.valid_dir)
    c['test_dir'] = os.path.join(c['base_dir'], FLAGS.test_dir)
    c['outdir_prefix'] = FLAGS.outdir_prefix
    c['log_dir'] = FLAGS.log_dir
    c['batch_size'] = FLAGS.batch_size
    c['H'] = FLAGS.H
    c['W'] = FLAGS.W
    c['sen1_Ch'] = FLAGS.sen1_Ch
    c['sen2_Ch'] = FLAGS.sen2_Ch
    c['label_class'] = FLAGS.label_class
    c['learning_rate'] = FLAGS.learning_rate
    c['regular_decay'] = FLAGS.regular_decay
    c['regular_weight_decay'] = FLAGS.regular_weight_decay
    c['lr_decay_steps'] = FLAGS.lr_decay_steps
    c['lr_decay'] = FLAGS.lr_decay
    c['epochs'] = FLAGS.epochs
    c['epoch_size'] = FLAGS.epoch_size
    c['min_learning_rate'] = FLAGS.min_learning_rate
    c['lr_decay_begin'] = FLAGS.lr_decay_begin
    c['patience'] = FLAGS.patience
    c['keep_prob'] = FLAGS.keep_prob
    if c['log_dir'] == None:
        log_dir = os.path.join(c['base_dir'], _generate_run_id(c))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        c['log_dir'] = log_dir
    return c

def load_data(config):
    path_train = config.get('train_dir')
    data = h5py.File(path_train, 'r')
    train_sen1 = np.array(data['sen1'])
    train_sen2 = np.array(data['sen2'])
    train_label = np.array(data['label'])
    path_valid = config.get('valid_dir')
    data = h5py.File(path_valid, 'r')
    valid_sen1 = np.array(data['sen1'])
    valid_sen2 = np.array(data['sen2'])
    valid_label = np.array(data['label'])
    path_test = config.get('test_dir')
    data = h5py.File(path_test, 'r')
    test_sen1 = np.array(data['sen1'])
    test_sen2 = np.array(data['sen2'])
    train_data = []
    train_data.append(train_sen1)
    train_data.append(train_sen2)
    train_data.append(train_label)
    valid_data = []
    valid_data.append(valid_sen1)
    valid_data.append(valid_sen2)
    valid_data.append(valid_label)
    test_data = []
    test_data.append(test_sen1)
    test_data.append(test_sen2)
    return train_data, valid_data, test_data

def split_batchsize_epochs(config, data, istest):
    batch_size = config.get('batch_size')
    H = config.get('H')
    W = config.get('W')
    sen1 = data[0]
    sen2 = data[1]
    samples = sen1.shape[0]
    epoch_size = samples // batch_size
    new_data = []
    if istest:
        sen1 = np.expand_dims(sen1, axis=1)
        sen2 = np.expand_dims(sen2, axis=1)
        new_data.append(sen1)
        new_data.append(sen2)
    else: # training and valid
        label = data[2]
        tot_size = epoch_size * batch_size
        sen1 = np.expand_dims(sen1[:tot_size, :, :, :], axis=1)
        sen2 = np.expand_dims(sen2[:tot_size, :, :, :], axis=1)
        label = np.expand_dims(label[:tot_size, :], axis=1)
        sen1 = np.reshape(sen1, newshape=(epoch_size, batch_size, H, W, -1))
        sen2 = np.reshape(sen2, newshape=(epoch_size, batch_size, H, W, -1))
        label = np.reshape(label, newshape=(epoch_size, batch_size, -1))
        new_data.append(sen1)
        new_data.append(sen2)
        new_data.append(label)
    return new_data

def distorted_inputs(config, logger):
    # Load whole dataset:
    start = time.time()
    train_data, valid_data, test_data = load_data(config)
    end = time.time()
    message = 'Load train_data(49G), valid_data(3.4G), test_data(0.7G): %ds' % (end - start)
    logger.info(message)
    # Split to batch_size data block:
    start = time.time()
    train_data = split_batchsize_epochs(config, train_data, False)
    valid_data = split_batchsize_epochs(config, valid_data, False)
    test_data = split_batchsize_epochs(config, test_data, True)
    end = time.time()
    message = 'Transform dataset to batched train_data, valid_data, test_data: %ds' % (end - start)
    logger.info(message)
    return train_data, valid_data, test_data

dtype = tf.float32
constinit = tf.constant_initializer
relu = tf.nn.relu
tanh = tf.nn.tanh
sigmoid = tf.nn.sigmoid
softmax = tf.nn.softmax
softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2
batch_normal = tf.layers.batch_normalization
MYNET_VARIABLES = 'mynet_variables'
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
UPDATE_OPS_COLLECTION = 'mynet_update_ops'  # must be grouped with training op

class Model:
    """
    myself design model class
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.log_dir = config['log_dir']
        self.keep_drop = tf.placeholder(dtype, (), name='dropout_kp')
        self.regular_decay = tf.placeholder(dtype, (), name='regular_decay')
        self.regular_weight_decay = tf.placeholder(dtype, (), name='regular_wdecay')
        self.lr = tf.get_variable('lr', (), initializer=constinit(config['learning_rate']), trainable=False)
        self.new_lr = tf.placeholder(dtype, (), name='new_lr')
        self._op_lr_update = tf.assign(self.lr, self.new_lr, name='lr_update')
        self._op_train = None
        self.loss = None
        H = config['H']
        W = config['W']
        sen1Ch = config['sen1_Ch']
        sen2Ch = config['sen2_Ch']
        lclass = config['label_class']
        self.in_sen1 = tf.placeholder(dtype, (None, H, W, sen1Ch), name='insen1')
        self.in_sen2 = tf.placeholder(dtype, (None, H, W, sen2Ch), name='insen2')
        self.label = tf.placeholder(dtype, (None, lclass), name='label')
        self.logit = None
        self.logit_aux1 = None
        self.logit_aux2 = None
        self.output = None
        self.accuracy = None
        self.global_step = tf.placeholder(tf.int32, (), name='my_globstep')
        self.is_training = tf.placeholder(tf.bool, (), name='is_training')
    def inference(self, use_bias=False):
        self.config['ksize'] = 7
        self.config['stride'] = 1
        self.config['use_bias'] = True

        with tf.variable_scope('sen1_1_1'):   # sen1: 1,2
            sen1_x_1 = self.in_sen1[..., :2]
            sen1_abs_1 = self.in_sen1[..., :2]
            sen1_1_complex = tf.complex(tf.expand_dims(sen1_abs_1[..., 0], axis=-1), tf.expand_dims(sen1_abs_1[..., 1], axis=-1))
            sen1_abs_1 = tf.abs(sen1_1_complex)
            sen1_angle_1 = tf.cast(tf.angle(sen1_1_complex), dtype=dtype)
            sen1_fft_1 = tf.fft3d(sen1_1_complex)
            sen1_fft_abs_1 = tf.abs(sen1_fft_1)
            sen1_fft_angle_1 = tf.cast(tf.angle(sen1_fft_1), dtype=dtype)
            sen1_x_1 = tf.concat([sen1_x_1, sen1_abs_1, sen1_angle_1, sen1_fft_abs_1, sen1_fft_angle_1], axis=-1)
            sen1_x_1 = batch_normal(sen1_x_1, training=self.is_training)
            with tf.variable_scope('block2'):
                sen1_x_1 = self.cnns_stack_block_adj(sen1_x_1)  # 16 * 16 * (256+4)=260
        with tf.variable_scope('sen1_1_2'):   # sen1: 3,4
            sen1_x_2 = self.in_sen1[..., 2:4]
            sen1_abs_2 = self.in_sen1[..., 2:4]
            sen1_2_complex = tf.complex(tf.expand_dims(sen1_abs_2[..., 0], axis=-1), tf.expand_dims(sen1_abs_2[..., 1], axis=-1))
            sen1_abs_2 = tf.abs(sen1_2_complex)
            sen1_angle_2 = tf.cast(tf.angle(sen1_2_complex), dtype=dtype)
            sen1_fft_2 = tf.fft3d(sen1_2_complex)
            sen1_fft_abs_2 = tf.abs(sen1_fft_2)
            sen1_fft_angle_2 = tf.cast(tf.angle(sen1_fft_2), dtype=dtype)
            sen1_x_2 = tf.concat([sen1_x_2, sen1_abs_2, sen1_angle_2, sen1_fft_abs_2, sen1_fft_angle_2], axis=-1)
            sen1_x_2 = batch_normal(sen1_x_2, training=self.is_training)
            with tf.variable_scope('block2'):
                sen1_x_2 = self.cnns_stack_block_adj(sen1_x_2)  # 16 * 16 * (256+4)=260
        with tf.variable_scope('sen1_1_3'):   # sen1: 5,6
            sen1_x_3 = self.in_sen1[..., 4:6]
            sen1_abs_3 = self.in_sen1[..., 4:6]
            sen1_3_complex = tf.complex(tf.expand_dims(sen1_abs_3[..., 0], axis=-1), tf.expand_dims(sen1_abs_3[..., 1], axis=-1))
            sen1_abs_3 = tf.abs(sen1_3_complex)
            sen1_angle_3 = tf.cast(tf.angle(sen1_3_complex), dtype=dtype)
            sen1_fft_3 = tf.fft3d(sen1_3_complex)
            sen1_fft_abs_3 = tf.abs(sen1_fft_3)
            sen1_fft_angle_3 = tf.cast(tf.angle(sen1_fft_3), dtype=dtype)
            sen1_x_3 = tf.concat([sen1_x_3, sen1_abs_3, sen1_angle_3, sen1_fft_abs_3, sen1_fft_angle_3], axis=-1)
            sen1_x_3 = batch_normal(sen1_x_3, training=self.is_training)
            with tf.variable_scope('block2'):
                sen1_x_3 = self.cnns_stack_block_adj(sen1_x_3)  # 16 * 16 * (256+2)=258
        with tf.variable_scope('sen1_1_4'):    # sen1: 7,8
            sen1_x_4 = self.in_sen1[...,6:]
            sen1_abs_4 = self.in_sen1[..., 6:]
            sen1_4_complex = tf.complex(tf.expand_dims(sen1_abs_4[..., 0], axis=-1), tf.expand_dims(sen1_abs_4[..., 1], axis=-1))
            sen1_abs_4 = tf.abs(sen1_4_complex)
            sen1_angle_4 = tf.cast(tf.angle(sen1_4_complex), dtype=dtype)
            sen1_fft_4 = tf.fft3d(sen1_4_complex)
            sen1_fft_abs_4 = tf.abs(sen1_fft_4)
            sen1_fft_angle_4 = tf.cast(tf.angle(sen1_fft_4), dtype=dtype)
            sen1_x_4 = tf.concat([sen1_x_4, sen1_abs_4, sen1_angle_4, sen1_fft_abs_4, sen1_fft_angle_4], axis=-1)
            sen1_x_4 = batch_normal(sen1_x_4, training=self.is_training)
            with tf.variable_scope('block2'):
                sen1_x_4 = self.cnns_stack_block_adj(sen1_x_4)  # 16 * 16 * (256+2)=258

        with tf.variable_scope('sen2_1_1'):   # sen2: 1,2,3
            sen2_x_1 = self.in_sen2[..., :3]
            sen2_x_1 = batch_normal(sen2_x_1, training=self.is_training)
            with tf.variable_scope('block2'):
                sen2_x_1 = self.cnns_stack_block_adj(sen2_x_1)  # 16 * 16 * 64
        with tf.variable_scope('sen2_1_2'):   # sen2: 4,5,6
            sen2_x_2 = self.in_sen2[..., 3:6]
            sen2_x_2 = batch_normal(sen2_x_2, training=self.is_training)
            with tf.variable_scope('block2'):
                sen2_x_2 = self.cnns_stack_block_adj(sen2_x_2)  # 16 * 16 * 64
        with tf.variable_scope('sen2_1_3'):   # sen2: 7,8
            sen2_x_3 = self.in_sen2[..., 6:8]
            sen2_x_3 = batch_normal(sen2_x_3, training=self.is_training)
            with tf.variable_scope('block2'):
                sen2_x_3 = self.cnns_stack_block_adj(sen2_x_3)  # 16 * 16 * 64
        with tf.variable_scope('sen2_1_4'):   # sen2: 9,10
            sen2_x_4 = self.in_sen2[..., 8:]
            sen2_x_4 = batch_normal(sen2_x_4, training=self.is_training)
            with tf.variable_scope('block2'):
                sen2_x_4 = self.cnns_stack_block_adj(sen2_x_4)  # 16 * 16 * 64

        sen1_x_1 = tf.concat([sen1_x_1, sen1_x_2, sen1_x_3, sen1_x_4], axis=-1)  # 16 * 16 * (260+258+258+259)=1035
        sen2_x_1 = tf.concat([sen2_x_1, sen2_x_2, sen2_x_3, sen2_x_4], axis=-1)  # 16 * 16 * 266

        with tf.variable_scope('sen1_2'):
            with tf.variable_scope('block1'):
                sen1_x = self.cnns_stack_block_adj_noscale(sen1_x_1)
            with tf.variable_scope('block2'):
                sen1_x = self.cnns_stack_block_adj(sen1_x)  # 8 * 8 * (512+1035)=1547
            with tf.variable_scope('block3'):
                sen1_x = self.cnns_stack_block_adj_noscale(sen1_x)
        with tf.variable_scope('sen2_2'):
            with tf.variable_scope('block1'):
                sen2_x = self.cnns_stack_block_adj_noscale(sen2_x_1)
            with tf.variable_scope('block2'):
                sen2_x = self.cnns_stack_block_adj(sen2_x)  # 8 * 8 * (512+266)=778
            with tf.variable_scope('block3'):
                sen2_x = self.cnns_stack_block_adj_noscale(sen2_x)

        x = tf.concat([sen1_x, sen2_x], axis=-1)  # 8 * 8 * (1547+778)=2325
        x_aux_2 = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool_aux2")  # 1 * 1 * 3349
        dims = x_aux_2.get_shape()[-1]
        x_aux_2 = tf.reshape(x_aux_2, shape=(-1, dims))
        with tf.variable_scope('fc1_aux2'):
            self.config['fc_units_out'] = 526
            x_aux_2 = self.fc(x_aux_2)
            x_aux_2 = relu(x_aux_2)
        with tf.variable_scope('fc2_aux2'):
            self.config['fc_units_out'] = 1024
            x_aux_2 = self.fc(x_aux_2)
            x_aux_2 = relu(x_aux_2)
        x_aux_2 = tf.nn.dropout(x_aux_2, self.keep_drop)
        with tf.variable_scope('fc_proj_aux2'):
            self.config['fc_units_out'] = self.config['label_class']
            x_aux_2 = self.fc(x_aux_2)
        self.logit_aux2 = x_aux_2

        with tf.variable_scope('hub_1'):
            with tf.variable_scope('block1'):
                x = self.cnns_stack_block_adj_noscale(x)
            with tf.variable_scope('block2'):
                x = self.cnns_stack_block_adj(x)   # 4 * 4 * (1024+2325)=3349
            with tf.variable_scope('block3'):
                x = self.cnns_stack_block_adj_noscale(x)
        x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")  # 1 * 1 * 3349
        dims = x.get_shape()[-1]
        x = tf.reshape(x, shape=(-1, dims))
        with tf.variable_scope('fc1'):
            self.config['fc_units_out'] = 1024
            x = self.fc(x)
            x = relu(x)
        x = tf.nn.dropout(x, self.keep_drop)
        with tf.variable_scope('fc_proj_hub'):
            self.config['fc_units_out'] = self.config['label_class']
            x = self.fc(x)
        self.logit = x
        self.output = softmax(x)
    def train_valid_algorithm_run(self):
        self.loss_cal()
        self.accuracy_cal()
        optimizer = tf.train.AdamOptimizer(self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(self.loss)
            self._op_train = train_op
    def train(self, sess, train_data, epoch_i, epoch_size):
        new_lr = 0.0
        lr_init = self.config['learning_rate']
        keep_prob = self.config['keep_prob']
        l2_decay = self.config['regular_decay']
        l2_wdecay = self.config['regular_weight_decay']
        lr_decay_internal = self.config['lr_decay_steps']
        lr_decay = self.config['lr_decay']
        min_learning_rate = self.config['min_learning_rate']
        lr_decay_begin = self.config['lr_decay_begin']
        losses = []
        fetches = {
            'train_op': self._op_train,
            'loss': self.loss
        }
        cur_global_step = None
        for i, (in_sen1, in_sen2, label) in enumerate(zip(train_data[0], train_data[1], train_data[2])):
            new_lr = max(min_learning_rate, lr_init * np.power(lr_decay, (max(0, i + epoch_i * epoch_size - lr_decay_begin))//lr_decay_internal))
            sess.run({'lr_assign_op': self._op_lr_update}, {self.new_lr: new_lr})
            feed_dict = {
                self.global_step: i + 1 + epoch_i * epoch_size,
                self.regular_decay: l2_decay,
                self.regular_weight_decay: l2_wdecay,
                self.keep_drop: keep_prob,
                self.is_training: True,
                self.in_sen1: in_sen1,
                self.in_sen2: in_sen2,
                self.label: label
            }
            vals = sess.run(fetches, feed_dict=feed_dict)
            losses.append(vals['loss'])
            cur_global_step = i + 1 + epoch_i * epoch_size
        results = {
            'loss': np.mean(losses),
            'global_step': cur_global_step
        }
        return results
    def eval(self, sess, valid_data, global_step):
        losses = []
        accuracys = []
        l2_decay = self.config['regular_decay']
        l2_wdecay = self.config['regular_weight_decay']
        fetches = {
            'loss': self.loss,
            'accuracy': self.accuracy
        }
        for _, (in_sen1, in_sen2, label) in enumerate(zip(valid_data[0], valid_data[1], valid_data[2])):
            feed_dict = {
                self.global_step: global_step,
                self.regular_decay: l2_decay,
                self.regular_weight_decay: l2_wdecay,
                self.keep_drop: 1.0,
                self.is_training: False,
                self.in_sen1: in_sen1,
                self.in_sen2: in_sen2,
                self.label: label
            }
            vals = sess.run(fetches, feed_dict=feed_dict)
            losses.append(vals['loss'])
            accuracys.append(vals['accuracy'])
        results = {
            'loss': np.mean(losses),
            'accuracy': np.mean(accuracys)
        }
        return results
    def test(self, sess, epoch_i, test_data, global_step):
        fetches = {
            'output': self.output
        }
        outputs = []
        for _, (in_sen1, in_sen2) in enumerate(zip(test_data[0], test_data[1])):
            feed_dict = {
                self.global_step: global_step,
                self.keep_drop: 1.0,
                self.is_training: False,
                self.in_sen1: in_sen1,
                self.in_sen2: in_sen2
            }
            vals = sess.run(fetches, feed_dict=feed_dict)
            output = np.zeros(shape=vals['output'].shape, dtype=np.int)
            output[0][np.argmax(vals['output'], axis=-1)] = 1
            outputs.append(output)
        outputs = np.concatenate(outputs, axis=0)
        # write to csv and save model:
        self.save_model(sess, global_step)
        result_csv = 'epoch%d_%d_result.csv' % (epoch_i, global_step)
        result_path = os.path.join(self.log_dir, result_csv)
        data = pd.DataFrame(outputs)
        data.to_csv(result_path,columns=None,header=False,index=False)
    def loss_cal(self):
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                           if 'bias' not in v.name]) * self.regular_weight_decay
        cross_entropy = softmax_cross_entropy(logits=self.logit, labels=self.label)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean
        cross_entropy_aux2 = softmax_cross_entropy(logits=self.logit_aux2, labels=self.label)
        cross_entropy_aux2_mean = tf.reduce_mean(cross_entropy_aux2)
        loss_aux2 = cross_entropy_aux2_mean
        self.loss = loss + loss_aux2 + lossL2 * self.regular_decay
        tf.summary.scalar('loss', self.loss)
    def accuracy_cal(self):
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype))
    def shuffle(self, x, groups):
        _, h, w, c = x.shape.as_list()
        x = tf.reshape(x, shape=(-1, h, w, groups, c // groups))
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        x = tf.reshape(x, shape=(-1, h, w, c))
        return x
    def cnns_stack_block_adj_noscale(self, x):
        # total 1 blocks-3cnns
        with tf.variable_scope('adj_block1'):
            self.config['conv_filters_out'] = 256
            self.config['ksize'] = 1
            self.config['stride'] = 1
            self.config['use_bias'] = True
            x_adj1 = self.conv(x)
            x_adj1 = relu(x_adj1)
        with tf.variable_scope('fc_block1'):
            self.config['conv_filters_out'] = 256
            self.config['ksize'] = 1
            self.config['stride'] = 1
            self.config['use_bias'] = True
            x_proj = self.conv(x)
            # x = batch_normal(x, training=self.is_training)
            x_proj = relu(x_proj)
        with tf.variable_scope('block1_size3'):
            self.config['conv_filters_out'] = 64
            self.config['ksize'] = 1
            self.config['stride'] = 1
            self.config['use_bias'] = True
            with tf.variable_scope('cnn1'):
                x1 = self.conv(x_proj)
                # x1 = batch_normal(x1, training=self.is_training)
                x1 = relu(x1)
            self.config['conv_filters_out'] = 64
            self.config['ksize'] = 3
            self.config['stride'] = 1
            self.config['use_bias'] = True
            with tf.variable_scope('cnn2'):
                x1 = self.conv(x1)
                # x1 = batch_normal(x1, training=self.is_training)
                x1 = relu(x1)
            self.config['conv_filters_out'] = 256
            self.config['ksize'] = 1
            self.config['stride'] = 1
            with tf.variable_scope('cnn3'):
                x1 = self.conv(x1)
                # x1 = batch_normal(x1, training=self.is_training)
                x1 = relu(x1 + x_proj)
        with tf.variable_scope('block2_size5'):
            self.config['conv_filters_out'] = 64
            self.config['ksize'] = 1
            self.config['stride'] = 1
            self.config['use_bias'] = True
            with tf.variable_scope('cnn1'):
                x2 = self.conv(x_proj)
                # x1 = batch_normal(x1, training=self.is_training)
                x2 = relu(x2)
            self.config['conv_filters_out'] = 64
            self.config['ksize'] = 5
            self.config['stride'] = 1
            self.config['use_bias'] = True
            with tf.variable_scope('cnn2'):
                x2 = self.conv(x2)
                # x1 = batch_normal(x1, training=self.is_training)
                x2 = relu(x2)
            self.config['conv_filters_out'] = 256
            self.config['ksize'] = 1
            self.config['stride'] = 1
            with tf.variable_scope('cnn3'):
                x2 = self.conv(x2)
                # x1 = batch_normal(x1, training=self.is_training)
                x2 = relu(x2 + x_proj)
        x1 = tf.concat([x1, x2, x_adj1], axis=-1)  # 1/1 * 1/1 * 1024
        return x1
    def cnns_stack_block_adj(self, x):
        # total 1 blocks-3cnns
        with tf.variable_scope('adj_block1'):
            x_adj1 = self.max_pool(x, 2, 2)
            self.config['conv_filters_out'] = 256
            self.config['ksize'] = 1
            self.config['stride'] = 1
            self.config['use_bias'] = True
            x_adj1 = self.conv(x_adj1)
            x_adj1 = relu(x_adj1)
        with tf.variable_scope('adj_block2'):
            self.config['conv_filters_out'] = 256
            self.config['ksize'] = 1
            self.config['stride'] = 2
            self.config['use_bias'] = True
            x_adj2 = self.conv(x)
            x_adj2 = relu(x_adj2)
        with tf.variable_scope('fc_block1'):
            self.config['conv_filters_out'] = 256
            self.config['ksize'] = 1
            self.config['stride'] = 2
            self.config['use_bias'] = True
            x_proj = self.conv(x)
            # x = batch_normal(x, training=self.is_training)
            x_proj = relu(x_proj)
        with tf.variable_scope('block1_size3'):
            self.config['conv_filters_out'] = 64
            self.config['ksize'] = 1
            self.config['stride'] = 1
            self.config['use_bias'] = True
            with tf.variable_scope('cnn1'):
                x1 = self.conv(x_proj)
                # x1 = batch_normal(x1, training=self.is_training)
                x1 = relu(x1)
            self.config['conv_filters_out'] = 64
            self.config['ksize'] = 3
            self.config['stride'] = 1
            self.config['use_bias'] = True
            with tf.variable_scope('cnn2'):
                x1 = self.conv(x1)
                # x1 = batch_normal(x1, training=self.is_training)
                x1 = relu(x1)
            self.config['conv_filters_out'] = 256
            self.config['ksize'] = 1
            self.config['stride'] = 1
            with tf.variable_scope('cnn3'):
                x1 = self.conv(x1)
                # x1 = batch_normal(x1, training=self.is_training)
                x1 = relu(x1 + x_proj)
        with tf.variable_scope('block2_size5'):
            self.config['conv_filters_out'] = 64
            self.config['ksize'] = 1
            self.config['stride'] = 1
            self.config['use_bias'] = True
            with tf.variable_scope('cnn1'):
                x2 = self.conv(x_proj)
                # x1 = batch_normal(x1, training=self.is_training)
                x2 = relu(x2)
            self.config['conv_filters_out'] = 64
            self.config['ksize'] = 5
            self.config['stride'] = 1
            self.config['use_bias'] = True
            with tf.variable_scope('cnn2'):
                x2 = self.conv(x2)
                # x1 = batch_normal(x1, training=self.is_training)
                x2 = relu(x2)
            self.config['conv_filters_out'] = 256
            self.config['ksize'] = 1
            self.config['stride'] = 1
            with tf.variable_scope('cnn3'):
                x2 = self.conv(x2)
                # x1 = batch_normal(x1, training=self.is_training)
                x2 = relu(x2 + x_proj)
        x1 = tf.concat([x1, x2, x_adj1, x_adj2], axis=-1)  # 1/2 * 1/2 * 1024
        return x1
    def conv(self, x):
        ksize = self.config['ksize']
        stride = self.config['stride']
        filters_out = self.config['conv_filters_out']
        filters_in = x.get_shape()[-1]
        shape = [ksize, ksize, filters_in, filters_out]
        initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
        weights = self._get_variable('weights',
                                shape=shape,
                                dtype=dtype,
                                initializer=tf.contrib.layers.xavier_initializer())
        if self.config['use_bias']:
            bias = self._get_variable('bias', filters_out, initializer=tf.zeros_initializer)
            return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME') + bias
        else:
            return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
    def max_pool(self, x, ksize=2, stride=2):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')
    def avg_pool(self, x, ksize=2, stride=2):
        return tf.nn.avg_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')
    def bn(self, x):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]
        if self.config['use_bias']:
            bias = self._get_variable('bias', params_shape, initializer=tf.zeros_initializer)
            return x + bias
        axis = list(range(len(x_shape) - 1))
        beta = self._get_variable('beta',
                             params_shape,
                             initializer=tf.zeros_initializer)
        gamma = self._get_variable('gamma',
                              params_shape,
                              initializer=tf.ones_initializer)
        moving_mean = self._get_variable('moving_mean',
                                    params_shape,
                                    initializer=tf.zeros_initializer,
                                    trainable=False)
        moving_variance = self._get_variable('moving_variance',
                                        params_shape,
                                        initializer=tf.ones_initializer,
                                        trainable=False)
        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
        mean, variance = tf.cond(self.is_training, lambda: (mean, variance), lambda: (moving_mean, moving_variance))
        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
        return x
    def fc(self, x):
        num_units_in = x.get_shape()[1]
        num_units_out = self.config['fc_units_out']
        weights_initializer = tf.truncated_normal_initializer(
            stddev=FC_WEIGHT_STDDEV)
        weights = self._get_variable('weights',
                                shape=[num_units_in, num_units_out],
                                initializer=tf.contrib.layers.xavier_initializer())
        biases = self._get_variable('bias',
                               shape=[num_units_out],
                               initializer=tf.zeros_initializer)
        x = tf.nn.xw_plus_b(x, weights, biases)
        return x
    def save_model(self, sess, global_step):
        saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join(self.config['log_dir'], 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=global_step)
    def get_total_trainable_parameter_size(self):
        """
        Calculates the total number of trainable parameters in the current graph.
        :return:
        """
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            total_parameters += np.product([x.value for x in variable.get_shape()])
        return total_parameters
    def _get_variable(self,
                      name,
                      shape,
                      initializer,
                      weight_decay=0.0,
                      dtype='float',
                      trainable=True):
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, MYNET_VARIABLES]
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype,
                               regularizer=regularizer,
                               collections=collections,
                               trainable=trainable)

def main():
    # fetch input config:
    c = fetch_global_config()
    # initial MyLogger:
    mylogger = get_logger(c['log_dir'])
    # preprocess samples(train, valid, test):
    train_data, valid_data, test_data = distorted_inputs(c, logger=mylogger)
    c['epoch_size'] = train_data[0].shape[0]
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        mymodel = Model(c, logger=mylogger)
        mymodel.inference(use_bias=False)
        mymodel.train_valid_algorithm_run()
        # Log model statistics.
        total_trainable_parameter = mymodel.get_total_trainable_parameter_size()
        mylogger.info('Total number of trainable parameters: %d' % total_trainable_parameter)
        for var in tf.global_variables():
            mylogger.debug('%s, %s' % (var.name, var.get_shape()))
        sess.run(tf.global_variables_initializer())
        patience_i = 0
        max_val_accuracy = float(0.0)
        epoch = 0
        epochs = c['epochs']
        epoch_size = c['epoch_size']
        patience = c['patience']
        while epoch <= epochs:
            start_time = time.time()
            train_results = mymodel.train(sess, train_data, epoch, epoch_size)
            global_step = train_results['global_step']
            valid_results = mymodel.eval(sess, valid_data, global_step)
            end_time = time.time()
            message = 'Epoch %d (%d) train_loss: %.6f, val_loss: %.6f, val_accuracy: %.5f %ds' % (
                    epoch, global_step, train_results['loss'], valid_results['loss'], valid_results['accuracy'], (end_time - start_time))
            mylogger.info(message)
            if valid_results['accuracy'] >= max_val_accuracy:
                patience_i = 0
                mylogger.info('Val accuracy improve from %.5f to %.5f' % (max_val_accuracy, valid_results['accuracy']))
                if epoch >= 0:
                    mymodel.test(sess, epoch, test_data, global_step)
                max_val_accuracy = valid_results['accuracy']
            else:
                patience_i += 1
                if patience_i > patience:
                    mylogger.warn('Early stopping at epoch: %d' % epoch)
                    break
            epoch += 1
            sys.stdout.flush()

if __name__ == '__main__':
    main()

