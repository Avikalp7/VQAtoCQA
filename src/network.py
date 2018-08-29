"""
Module implementing wrappers over tensorflow
"""
# Copyright (C) 2018 Yahoo Japan Corporation (Licensed under CC BY-NC-SA 4.0)
# https://creativecommons.org/licenses/by-nc-sa/4.0/


from math import sqrt

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from global_hyperparams import WEIGHT_DECAY

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    """A little wrapper around tf.get_variable to do weight decay and add to resnet collection"""
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           trainable=trainable)


def weight_variable(is_trainable, shape, stddev=None, initializer_type='normal', name=''):
    if stddev is None:
        if len(shape) == 4:
            stddev = 1 / sqrt(shape[0] * shape[1] * shape[2])
        elif len(shape) == 2:
            stddev = 1 / sqrt(shape[0])
        else:
            stddev = 0.1

    # initial = tf.truncated_normal(shape, stddev=stddev)
    if initializer_type == 'normal':
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    elif initializer_type == 'xavier':
        initializer = xavier_initializer()
    else:
        raise ValueError("Unknown initializer type %s for weight variable" % initializer_type)

    return _get_variable(name='weight-' + name, shape=shape, initializer=initializer, weight_decay=WEIGHT_DECAY,
                         trainable=is_trainable)


def bias_variable(is_trainable, shape, name=''):
    # initial = tf.constant(0.1, shape=shape)
    initializer = tf.constant_initializer(value=0.1)
    return _get_variable(name='bias_var-' + name, shape=shape, initializer=initializer, weight_decay=0.0,
                         trainable=is_trainable)
    # return tf.Variable(initial, name='bias_var')


def conv2d(x, W, strides=None, padding='SAME', isnorm=False, is_training=False):
    """
    Prepare a tf.nn.conv2d layer and return it
    :param x: 4-D tensor, input to conv2d layer
    :param W: 4-D tensor, filter for conv2d layer
    :param strides: list of 4 ints, stride over batch, height, width, channel
    :param padding: enum, 'SAME' or 'VALID'
    :param isnorm: bool, whether to apply batch normalization
    :param is_training: bool, whether in training mode.
    :return: tf.nn.conv2d object, with x as input, W as filter
    """
    if strides is None:
        strides = [1, 1, 1, 1]
    if isnorm:
        x = tf.contrib.layers.batch_norm(inputs=x, is_training=is_training)
        # x = tf_nn_batch_norm(input_tensor=x)
    return tf.nn.conv2d(input=x, filter=W, strides=strides, padding=padding)


def batch_norm_conv_activation(inputs, is_training, activation, is_trainable):
    """Performs a batch normalization followed by activation."""
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True, trainable=is_trainable)
    if activation is not None:
        inputs = activation(inputs)
    return inputs


def batch_norm_dense_activation(inputs, is_training, activation, is_trainable):
    """Perform a batch normalization followed by activation.

    Parameters
    ----------
    :param inputs: tensor, of shape [batch_size, layer_dimension]

    :param is_training: bool, whether the model is in training mode

    :param activation: one of TF's activation function

    :param is_trainable: bool, whether the model is trainable

    Returns
    -------
    bn_inputs: tensor, 'inputs' with batch norm and activation applied to it
    """
    bn_inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, trainable=is_trainable)
    bn_inputs = activation(bn_inputs)
    return bn_inputs


def standard_FC_layer(inputs, input_dim, output_dim, use_batch_norm, activation, train_mode, is_trainable,
                      dropout_keep_prob, name_suffix):
    initializer_type = get_initializer_type(activation)
    W_I = weight_variable(is_trainable=is_trainable, shape=[input_dim, output_dim],
                          initializer_type=initializer_type, name='W_%s' % name_suffix)
    b_I = bias_variable(is_trainable=is_trainable, shape=[output_dim], name='b_%s' % name_suffix)
    if use_batch_norm:
        output = batch_norm_dense_activation(inputs=tf.nn.xw_plus_b(x=inputs,
                                                                    weights=W_I,
                                                                    biases=b_I),
                                             activation=activation,
                                             is_training=train_mode,
                                             is_trainable=is_trainable)
    else:
        output = activation(tf.nn.xw_plus_b(x=inputs, weights=W_I, biases=b_I))

    output = tf.nn.dropout(output, dropout_keep_prob)
    return output


def standard_conv_layer(inputs, filter_shape, activation, use_batch_norm, train_mode, is_trainable, name_suffix):
    initializer_type = get_initializer_type(activation)
    W_conv1 = weight_variable(is_trainable=is_trainable, shape=filter_shape,
                              initializer_type=initializer_type,
                              name='W_%s' % name_suffix)
    b_conv1 = bias_variable(is_trainable=is_trainable, shape=[512], name="b_%s" % name_suffix)
    if not use_batch_norm:
        conv_output = activation(conv2d(x=inputs, W=W_conv1) + b_conv1, name='h_%s' % name_suffix)
    else:
        conv_output = batch_norm_conv_activation(inputs=conv2d(x=inputs, W=W_conv1) + b_conv1,
                                                 is_training=train_mode,
                                                 activation=activation, is_trainable=is_trainable)
    return conv_output


def get_initializer_type(activation):
    return "normal" if activation == tf.nn.relu else "xavier"


def tf_nn_batch_norm(input_tensor):
    mean, variance = tf.nn.moments(input_tensor, [0, 1, 2])
    return tf.nn.batch_normalization(input_tensor, mean, variance, None, None, 1e-5)


def max_pool(x, ksize=None, strides=None, padding='SAME'):
    if ksize is None:
        ksize = [1, 2, 2, 1]
    if strides is None:
        strides = [1, 2, 2, 1]
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)


# local_response_normalization
def lr_norm(x, depth_radius, bias=1.0, alpha=0.001 / 9.0, beta=0.75):
    return tf.nn.lrn(x, depth_radius, bias=bias, alpha=alpha, beta=beta)
