import math

import tensorflow as tf
import numpy as np

tf.set_random_seed(20160905)

"""
Helper class for quickly creating NN layers with different initializations
"""


def att_w(batch_dim=128, input_dim=100, output_dim=100, name='W_att', init='Xavier', reg=None):
    return variable([batch_dim, input_dim, output_dim], name, init, reg)


def ff_w(input_dim=100, output_dim=100, name='W', init='Xavier', reg=None):
    return variable([input_dim, output_dim], name, init, reg)


def ff_b(dim=100, name='B', init='Zero', reg=None):
    return variable([dim], name, init, reg)


def conv_w(depth=None, width=3, height=3, in_channels=20, out_channels=100,
           name='Conv', init='Xavier', reg=None):
    if depth is None:
        return variable([height, width, in_channels, out_channels], name, init, reg)
    return variable([depth, height, width, in_channels, out_channels], name, init, reg)


def lstm_cell(hidden_dim=100):
    return tf.nn.rnn_cell.LSTMCell(hidden_dim, state_is_tuple=True,
                                   initializer=tf.contrib.layers.xavier_initializer(
                                       seed=20160501))

# Courtesy of Kevin Shih
def batch_norm_layer(x,train_phase,scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=0.99, center=False, scale=False,
                                            is_training=True,
                                            reuse=None,
                                            trainable=True,
                                            #updates_collections=None,
                                            scope=scope_bn)
    bn_inference = tf.contrib.layers.batch_norm(x, decay=0.99, center=False, scale=False,
                                                is_training=False,
                                                reuse=True,
                                                trainable=True,
                                                #updates_collections=None,
                                                scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z


def variable(shape, name, init='Xavier', reg=None):
    with tf.variable_scope(name) as scope:
        regularizer = None
        if reg == 'l2':
            regularizer = tf.contrib.layers.l2_regularizer(1e-3)
        elif reg == 'l1':
            regularizer = tf.contrib.layers.l1_regularizer(1e-4)

        if init == 'Zero':
            vals = np.zeros(shape).astype('f')
            return tf.get_variable(name=name, initializer=vals, dtype=tf.float32)
        if init == 'One':
            vals = np.ones(shape).astype('f')
            return tf.get_variable(name=name, initializer=vals, dtype=tf.float32)

        if init == 'Xavier':
            w_init = tf.contrib.layers.xavier_initializer(seed=20161016)
        elif init == 'Normal':
            w_init = tf.random_normal_initializer(stddev=1.0 / math.sqrt(shape[0]),
                                                  seed=12132015)
        elif init == 'Uniform':
            w_init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05,
                                                   seed=12132015)
        return tf.get_variable(name=name, shape=shape, initializer=w_init, regularizer=regularizer)