"""Module implementing image-based models for category classification.

The only model from this module used in our paper is ResNet.

TODO: Add more documentation in this module
"""
# Copyright (C) 2018 Yahoo Japan Corporation (Licensed under CC BY-NC-SA 4.0)
# https://creativecommons.org/licenses/by-nc-sa/4.0/


import os

# Elevating TF log level to remove tensorflow CPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import resnet.resnet2 as resnet
from base_model import BaseModel
from utils import update_train_batch, update_val_batch, \
    update_test_batch, training_epoch_finish_routine, test_finish_routine
from paths import get_resnet_stored_filename
from network import weight_variable, conv2d, lr_norm, max_pool, bias_variable, \
    batch_norm_conv_activation, batch_norm_dense_activation
from global_hyperparams import NUM_CLASSES, ModelType, ModelName, training_epochs_dict, batch_size_dict, \
    RESNET_LAYERS, use_batch_norm, IMAGE_SIZE, IMAGE_PIXELS


class ImageModel(BaseModel):
    """Base model class for all image based models

    Implements a common train and test routine for all, along with tensorboard visualization
    """

    def __init__(self, model_name, model_type=None, is_trainable=True, is_primary_model=True):
        if model_type is None:
            super(ImageModel, self).__init__(model_type=ModelType.image_only, model_name=model_name,
                                             is_trainable=is_trainable, is_primary_model=is_primary_model)
        else:
            super(ImageModel, self).__init__(model_type=model_type, model_name=model_name,
                                             is_trainable=is_trainable, is_primary_model=is_primary_model)

        # placeholder for image data
        self.images_placeholder = tf.placeholder("float", shape=[None, IMAGE_PIXELS], name='images')

        # input layer # 4D tensor: [batch_size, image_size, image_size, channel_size]
        self.x_image = tf.reshape(self.images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    def save_bottleneck_data(self, sess, layer_name, input_ids, input_labels, input_texts, output_filename):
        raise NotImplementedError

    def _train_helper_SGD(self, sess, train_writer, train_ids, train_labels, train_texts,
                          val_ids, val_labels, val_texts, iterator_train_init, next_element_train, iterator_val_init,
                          next_element_val):
        # Ignore texts
        batch_size = batch_size_dict[self.model_type]
        training_epochs = training_epochs_dict[self.model_type]

        # EARLY STOPPING VARS
        val_acc_history = []
        epochs_with_current_lrate = 0
        min_increment = 0.0002
        tolerable_decrease = 0.0004

        global_step = 0
        for epoch in range(training_epochs):
            # Decay learning rate if required
            early_stop, epochs_with_current_lrate = self.early_stopping_procedure(sess,
                                                                                  epochs_with_current_lrate,
                                                                                  val_acc_history,
                                                                                  min_increment, tolerable_decrease)
            if early_stop: break

            # Running over full training data
            try:
                print("\nCurrent learning rate: {}".format(sess.run(self.learning_rate)))
            except TypeError:
                print("\nCurrent learning rate: {}".format(self.learning_rate))
            # Running training GD for this epoch
            global_step = self._train_SGD_loop(sess, iterator_train_init, next_element_train,
                                               train_labels, None, train_writer, global_step)

            # Now performing validation
            num_val_samples, val_acc_sum = self._val_prediction_loop(sess, iterator_val_init,
                                                                     next_element_val, val_labels,
                                                                     None, epoch)

            training_epoch_finish_routine(sess, val_acc_sum, num_val_samples, self.train_logfile_name,
                                          self.checkpoint_dir, epoch, self.saver)

        # Completed all epochs
        print("Completed training %s model, with %d epochs, each with %d minibatches each of size %d"
              % (self.model_name.name, training_epochs + 1, global_step + 1, batch_size))

    def _train_SGD_loop(self, sess, iterator_train_init, next_element_train, train_labels, text_train,
                        train_writer, global_step):
        """Run the gradient descent loop over batches for this epoch"""
        # Initialize the iterator
        sess.run(iterator_train_init)
        # Initialize other book-keeping variables
        batch_size = batch_size_dict[self.model_type]
        total_batches = int(len(train_labels) / batch_size) + 1
        batches_completed_this_epoch = 0
        total_epoch_acc_val = 0.
        print("*** TOTAL BATCHES: %d ***" % total_batches)
        # Running over all mini-batches in this epoch
        while True:
            try:
                batch_image, _, batch_ids = sess.run(next_element_train)
                batches_completed_this_epoch += 1
            # Completed all minibatches in train set
            except tf.errors.OutOfRangeError:
                break

            S, loss_val, acc_val = self._train_SGD_batch_step(sess, batch_image, train_labels[batch_ids],
                                                              batch_ids, text_train, batch_step=global_step)
            global_step, total_epoch_acc_val = update_train_batch(global_step, S, loss_val, acc_val, train_writer,
                                                                  total_epoch_acc_val, batches_completed_this_epoch)

        return global_step

    def _val_prediction_loop(self, sess, iterator_val_init, next_element_val, val_labels, text_val, epoch):
        """Run prediction over all batches in the validation set"""
        print("\nRunning on validation data for epoch number %d" % (epoch + 1))

        batch_size = batch_size_dict[self.model_type]
        total_batches = int(len(val_labels) / batch_size) + 1

        # Initialize the iterator and other book-keeping vars
        sess.run(iterator_val_init)
        val_acc_sum = 0
        num_steps = 0
        num_val_samples = 0
        while True:
            try:
                batch_image, _, batch_ids = sess.run(next_element_val)
            # Completed all minibatches in validation set
            except tf.errors.OutOfRangeError:
                break

            val_acc = self._validation_batch_step(sess, batch_image, val_labels[batch_ids], batch_ids, text_val,
                                                  batch_step=num_steps)
            val_acc_sum, num_steps = update_val_batch(num_steps, val_acc, val_acc_sum, total_batches)
            num_val_samples += len(batch_ids)
        return num_val_samples, val_acc_sum

    def _run_tests(self, sess, test_ids, test_labels, test_texts, iterator_test_init, next_element_test):
        # Ignore texts
        _ = test_texts
        print("\nRunning test data through model ...")
        # Some required variables
        conf_matrix = None
        prediction = None
        test_accuracy = 0.
        c_mat = tf.contrib.metrics.confusion_matrix(self.predictions, tf.argmax(self.labels_placeholder, 1),
                                                    num_classes=NUM_CLASSES)
        total_batches = (len(test_labels) / batch_size_dict[self.model_type]) + 1
        sess.run(iterator_test_init)
        batch_num = 0
        while True:
            try:
                batch_image, _, batch_ids = sess.run(next_element_test)
            except tf.errors.OutOfRangeError:
                # Completed all minibatches in test set
                break

            pred, mat, test_acc = self._test_batch_step(sess, c_mat,
                                                        batch_image, test_labels[batch_ids], batch_ids, text_test=None)

            conf_matrix, prediction, test_accuracy = \
                update_test_batch(batch_num, pred, mat, test_acc, conf_matrix, prediction, test_accuracy, total_batches)
            batch_num += 1

        test_accuracy /= float(len(test_labels))
        labels, results = test_finish_routine(test_labels, prediction)

        return test_accuracy, labels, results, conf_matrix

    def _run_validation_test(self, sess, val_ids, val_labels, val_texts, iterator_validation_init, next_element_val):
        batch_size = batch_size_dict[self.model_type]
        total_batches = int(len(val_labels) / batch_size) + 1

        sess.run(iterator_validation_init)

        val_acc_sum = 0
        num_steps = 0
        while True:
            try:
                batch_image, batch_label, batch_ids = sess.run(next_element_val)
            except tf.errors.OutOfRangeError:
                # Completed all minibatches in test set
                break
            val_acc = self._validation_batch_step(sess, batch_image, batch_label, batch_ids, None, batch_step=num_steps)
            val_acc_sum, num_steps = update_val_batch(num_steps, val_acc, val_acc_sum, total_batches)

        num_samples = len(val_ids)
        print("*** Validation accuracy: %0.4f ***" % (val_acc_sum / float(num_samples)))

    def _train_SGD_batch_step(self, sess, batch_image, batch_label, batch_ids, text_train, train_mode=True,
                              batch_step=1):
        # ignore batch_ids & text_train
        _, _ = batch_ids, text_train
        S, _, loss_val, acc_val = sess.run(
            [self.train_summary_op, self.train_op, self.loss, self.accuracy],
            feed_dict={
                self.images_placeholder: batch_image,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 0.5,
                self.train_mode: train_mode
            })
        return S, loss_val, acc_val

    def _validation_batch_step(self, sess, batch_image, batch_label, batch_ids, text_val, train_mode=True,
                               batch_step=1):
        # ignore batch_ids & text_train
        _ = batch_ids, text_val, batch_step
        val_acc_step = sess.run(
            self.sum_accuracy,
            feed_dict={
                self.images_placeholder: batch_image,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 1.0,
                self.train_mode: train_mode
            })
        return val_acc_step

    def _test_batch_step(self, sess, c_mat, batch_image, batch_label, batch_ids, text_test, train_mode=True):
        # ignore batch_ids & text_test
        _, _ = batch_ids, text_test
        pred, mat, test_acc = sess.run(
            [self.probabilities, c_mat, self.sum_accuracy],
            feed_dict={
                self.images_placeholder: batch_image,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 1.0,
                self.train_mode: train_mode
            })
        return pred, mat, test_acc


class Resnet(ImageModel):
    def __init__(self, train_last_block, restore=False, model_name=None, is_trainable=False):
        if model_name is None:
            super(Resnet, self).__init__(model_name=ModelName.resnet, is_trainable=is_trainable)
        else:
            super(Resnet, self).__init__(model_name=model_name, is_trainable=is_trainable)

        self.bottleneck = True
        dim_m = int(IMAGE_SIZE / 32) * int(IMAGE_SIZE / 32)
        dim_d = 512 if not self.bottleneck else 512 * 4

        if RESNET_LAYERS == 50:
            num_blocks = [3, 4, 6, 3]
        elif RESNET_LAYERS == 101:
            num_blocks = [3, 4, 23, 3]
        elif RESNET_LAYERS == 152:
            num_blocks = [3, 8, 36, 3]
        else:
            raise ValueError("Invalid value %d for RESNET_LAYERS" % RESNET_LAYERS)

        _, self.avg_pool_representation, self.representation, self.representation_one_block_before = resnet.inference(
            self.images_placeholder,
            num_classes=1000,
            is_training=self.train_mode,
            bottleneck=self.bottleneck,
            num_blocks=num_blocks,
            train_last_block=train_last_block)

        self.representation = tf.reshape(self.representation, shape=[-1, dim_m * dim_d])

        if restore:
            resnet_ckpt_filename = get_resnet_stored_filename(file_type='ckpt', num_layers=RESNET_LAYERS)
            self.global_variable_init = tf.global_variables_initializer()
            vars_to_restore = tf.global_variables()
            saver = tf.train.Saver(vars_to_restore)
            with tf.Session() as resnet_sess:
                self.restore_model_from_filename(resnet_sess, model_filename=resnet_ckpt_filename, saver=saver)

        print("Completed Resnet.")

    def save_bottleneck_data(self, sess, layer_name, input_ids, input_labels, input_texts, output_filename):
        raise NotImplementedError


class ImageCNN(ImageModel):
    """
    A baseline convolution-pool based CNN model for image based QA classification.
    Based on Tamaki-san's code
    """

    def __init__(self, is_trainable=True):
        super(ImageCNN, self).__init__(model_name=ModelName.image_cnn)

        # MODEL DEFINITION
        # ================================================================================
        # conv 1 # padding='SAME'
        with tf.variable_scope('image_conv1'):
            # [filter_size, filter_size, channel_size, num_filters]
            self.W_conv1 = weight_variable(is_trainable=is_trainable, shape=[7, 7, 3, 64], stddev=5e-2)
            self.b_conv1 = bias_variable(is_trainable=is_trainable, shape=[64])
            self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1, is_training=self.train_mode) + self.b_conv1)
        # [128,128,64] -> [64,64,64] # ksize=[1,2,2,1], strides=[1,2,2,1]
        with tf.variable_scope('image_pool1'):
            self.h_pool1 = max_pool(self.h_conv1)

        # conv2
        with tf.variable_scope('image_conv2'):
            self.W_conv2 = weight_variable(is_trainable=is_trainable, shape=[7, 7, 64, 128], stddev=5e-2)
            self.b_conv2 = bias_variable(is_trainable=is_trainable, shape=[128])
            self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2, isnorm=False,
                                             is_training=self.train_mode) + self.b_conv2)
        # [64,64,128] -> [32,32,128]
        with tf.variable_scope('image_pool2'):
            self.h_pool2 = max_pool(self.h_conv2)

        # conv 3 [32,32,128] -> [16,16,256]
        with tf.variable_scope('image_conv3'):
            self.W_conv3 = weight_variable(is_trainable=is_trainable, shape=[5, 5, 128, 256], stddev=5e-2)
            self.b_conv3 = bias_variable(is_trainable=is_trainable, shape=[256])
            self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3, strides=[1, 2, 2, 1],
                                             isnorm=False, is_training=self.train_mode) + self.b_conv3)
        # [16,16,256] -> [8,8,256]
        with tf.variable_scope('image_pool3'):
            self.h_pool3 = max_pool(self.h_conv3)
        # norm 3
        with tf.variable_scope('image_norm3'):
            self.h_norm3 = lr_norm(self.h_pool3, 4)

        # conv 4 [8,8,256] -> [2,2,512]
        with tf.variable_scope('image_conv4'):
            self.W_conv4 = weight_variable(is_trainable=is_trainable, shape=[5, 5, 256, 512], stddev=5e-2)
            self.b_conv4 = bias_variable(is_trainable=is_trainable, shape=[512])
            self.h_conv4 = tf.nn.relu(conv2d(self.h_norm3, self.W_conv4, strides=[1, 4, 4, 1],
                                             isnorm=False, is_training=self.train_mode) + self.b_conv4)
        # [2,2,512] -> [1,1,512]
        with tf.variable_scope('image_pool4'):
            self.h_pool4 = max_pool(self.h_conv4)

        # fc
        with tf.variable_scope('image_fc1'):
            self.W_fc1 = weight_variable(is_trainable=is_trainable, shape=[512, 256], stddev=0.04)
            self.b_fc1 = bias_variable(is_trainable=is_trainable, shape=[256])
            self.h_pool4_flat = tf.reshape(self.h_pool4, [-1, 512])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool4_flat, self.W_fc1) + self.b_fc1)
            # dropout
            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.dropout_keep_prob)
        # softmax
        with tf.variable_scope('softmax'):
            self.W_softmax = weight_variable(is_trainable=is_trainable, shape=[256, NUM_CLASSES], stddev=0.01)
            self.b_softmax = bias_variable(is_trainable=is_trainable, shape=[NUM_CLASSES])
            self.probabilities = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_softmax) + self.b_softmax)

        # Finalize the predictions, the optimizing function, loss/accuracy stats etc.
        self._set_predictions_optimizer_and_loss()

    def save_bottleneck_data(self, sess, layer_name, input_ids, input_labels, input_texts, output_filename):
        raise NotImplementedError


class ImageCNNV2(ImageModel):
    """ A baseline convolution-pool based CNN model for image based QA classification. """

    def __init__(self, is_primary_model=True, is_trainable=True, train_last_layers=False):
        model_name = ModelName.image_cnn_v2
        # with tf.variable_scope(model_name.name):
        super(ImageCNNV2, self).__init__(model_name=model_name, is_primary_model=is_primary_model)

        last_pool_image_dim = int(IMAGE_SIZE / 32)

        # MODEL CONFIG:
        self.config = {
            'USE_AVG_POOLING': True,
            'IMAGE_SIZE': IMAGE_SIZE,
            'activation': tf.nn.relu,
        }

        activation = self.config['activation']

        # MODEL DEFINITION
        # ================================================================================
        # conv 1 # padding='SAME'
        # [IMG,IMG,3] -> [IMG,IMG,32]
        with tf.variable_scope('image_conv1'):
            # [filter_size, filter_size, channel_size, num_filters]
            W_conv1 = weight_variable(is_trainable=is_trainable, shape=[7, 7, 3, 32])
            b_conv1 = bias_variable(is_trainable=is_trainable, shape=[32])
            if use_batch_norm:
                self.h_conv1 = batch_norm_conv_activation(is_trainable=is_trainable,
                                                          inputs=conv2d(x=self.x_image, W=W_conv1) + b_conv1,
                                                          is_training=self.train_mode, activation=activation)
            else:
                self.h_conv1 = activation(conv2d(x=self.x_image, W=W_conv1) + b_conv1)

        # conv1-pool 1
        # [IMG,IMG,32] -> [IMG/2,IMG/2,32] # ksize=[1,2,2,1], strides=[1,2,2,1]
        with tf.variable_scope('image_pool1'):
            self.h_pool1 = max_pool(self.h_conv1)
        # # norm
        # with tf.variable_scope('image_norm1'):
        #     h_norm1 = lr_norm(self.h_pool1, 4)

        # conv2
        # [IMG/2,IMG/2,32] -> [IMG/2,IMG/2,64]
        with tf.variable_scope('image_conv2'):
            # [filter_size, filter_size, channel_size, num_filters]
            W_conv2 = weight_variable(is_trainable=is_trainable, shape=[5, 5, 32, 64])
            b_conv2 = bias_variable(is_trainable=is_trainable, shape=[64])
            if use_batch_norm:
                self.h_conv2 = batch_norm_conv_activation(is_trainable=is_trainable,
                                                          inputs=conv2d(x=self.h_pool1, W=W_conv2) + b_conv2,
                                                          is_training=self.train_mode, activation=activation)
            else:
                self.h_conv2 = activation(conv2d(x=self.h_pool1, W=W_conv2) + b_conv2)

        # conv3
        # [IMG/2,IMG/2,64] -> [IMG/2,IMG/2,64]
        with tf.variable_scope('image_conv3'):
            # [filter_size, filter_size, channel_size, num_filters]
            W_conv3 = weight_variable(is_trainable=is_trainable, shape=[5, 5, 64, 64])
            b_conv3 = bias_variable(is_trainable=is_trainable, shape=[64])
            if use_batch_norm:
                self.h_conv3 = batch_norm_conv_activation(is_trainable=is_trainable,
                                                          inputs=conv2d(x=self.h_conv2, W=W_conv3) + b_conv3,
                                                          is_training=self.train_mode, activation=activation)
            else:
                self.h_conv3 = activation(conv2d(x=self.h_conv2, W=W_conv3) + b_conv3)

        # conv3-pool2
        # [IMG/2,IMG/2,64] -> [IMG/4,IMG/4,64]
        with tf.variable_scope('image_pool2'):
            self.h_pool2 = max_pool(self.h_conv3)

        # conv4
        # [IMG/4,IMG/4,64] -> [IMG/4,IMG/4,128]
        with tf.variable_scope('image_conv4'):
            # [filter_size, filter_size, channel_size, num_filters]
            W_conv4 = weight_variable(is_trainable=is_trainable, shape=[3, 3, 64, 128])
            b_conv4 = bias_variable(is_trainable=is_trainable, shape=[128])
            if use_batch_norm:
                self.h_conv4 = batch_norm_conv_activation(is_trainable=is_trainable,
                                                          inputs=conv2d(x=self.h_pool2, W=W_conv4) + b_conv4,
                                                          is_training=self.train_mode, activation=activation)
            else:
                self.h_conv4 = activation(conv2d(x=self.h_pool2, W=W_conv4) + b_conv4)

        # conv5
        # [IMG/4,IMG/4,128] -> [IMG/4,IMG/4,128]
        with tf.variable_scope('image_conv5'):
            # [filter_size, filter_size, channel_size, num_filters]
            W_conv5 = weight_variable(is_trainable=is_trainable, shape=[3, 3, 128, 128])
            b_conv5 = bias_variable(is_trainable=is_trainable, shape=[128])
            if use_batch_norm:
                self.h_conv5 = batch_norm_conv_activation(is_trainable=is_trainable,
                                                          inputs=conv2d(x=self.h_conv4, W=W_conv5) + b_conv5,
                                                          is_training=self.train_mode, activation=activation)
            else:
                self.h_conv5 = activation(conv2d(x=self.h_conv4, W=W_conv5) + b_conv5)

        # conv5-pool3
        # [IMG/4,IMG/4,128] -> [IMG/8,IMG/8,128]
        with tf.variable_scope('image_pool3'):
            self.h_pool3 = max_pool(self.h_conv5)

        last_layers_trainable = is_trainable or train_last_layers

        # conv6
        # [IMG/8,IMG/8,128] -> [IMG/8,IMG/8,256]
        with tf.variable_scope('image_conv6'):
            # [filter_size, filter_size, channel_size, num_filters]
            W_conv6 = weight_variable(is_trainable=is_trainable, shape=[3, 3, 128, 256])
            b_conv6 = bias_variable(is_trainable=is_trainable, shape=[256])
            if use_batch_norm:
                self.h_conv6 = batch_norm_conv_activation(is_trainable=last_layers_trainable,
                                                          inputs=conv2d(x=self.h_pool3, W=W_conv6) + b_conv6,
                                                          is_training=self.train_mode, activation=activation)
            else:
                self.h_conv6 = activation(conv2d(x=self.h_pool3, W=W_conv6) + b_conv6)

        # conv7
        # [IMG/8,IMG/8,256] -> [IMG/8,IMG/8,256]
        with tf.variable_scope('image_conv7'):
            # [filter_size, filter_size, channel_size, num_filters]
            W_conv7 = weight_variable(is_trainable=is_trainable, shape=[3, 3, 256, 256])
            b_conv7 = bias_variable(is_trainable=is_trainable, shape=[256])
            if use_batch_norm:
                self.h_conv7 = batch_norm_conv_activation(is_trainable=last_layers_trainable,
                                                          inputs=conv2d(x=self.h_conv6, W=W_conv7) + b_conv7,
                                                          is_training=self.train_mode, activation=activation)
            else:
                self.h_conv7 = activation(conv2d(x=self.h_conv6, W=W_conv7) + b_conv7)

        # conv7-pool4
        # [IMG/8,IMG/8,256] -> [IMG/16,IMG/16,256]
        with tf.variable_scope('image_pool4'):
            self.h_pool4 = max_pool(self.h_conv7)

        # conv8
        # [IMG/16,IMG/16,256] -> [IMG/16,IMG/16,512]
        with tf.variable_scope('image_conv8'):
            # [filter_size, filter_size, channel_size, num_filters]
            W_conv8 = weight_variable(is_trainable=last_layers_trainable, shape=[3, 3, 256, 512])
            b_conv8 = bias_variable(is_trainable=last_layers_trainable, shape=[512])
            if use_batch_norm:
                self.h_conv8 = batch_norm_conv_activation(is_trainable=last_layers_trainable,
                                                          inputs=conv2d(x=self.h_pool4, W=W_conv8) + b_conv8,
                                                          is_training=self.train_mode, activation=activation)
            else:
                self.h_conv8 = activation(conv2d(x=self.h_pool4, W=W_conv8) + b_conv8)

        if self.config['USE_AVG_POOLING']:
            # conv8-avgPool
            # [IMG/16, IMG/16, 512] -> [512]
            with tf.variable_scope('image_avg_pool'):
                self.h_pool5_flat = tf.reduce_mean(self.h_conv8, reduction_indices=[1, 2], name="avg_pool")
        else:
            # conv8-pool5
            # [IMG/16,IMG/16,512] -> [IMG/32,IMG/32,512]
            with tf.variable_scope('image_pool5'):
                self.h_pool5 = max_pool(self.h_conv8)
                # Flatten last pool layer
                self.h_pool5_flat = tf.reshape(self.h_pool5,
                                               shape=[-1, last_pool_image_dim * last_pool_image_dim * 512],
                                               name='h_pool5_flat')

        if not self.config['USE_AVG_POOLING']:
            # FC0 [image_dim*image_dim*512] -> [512]
            with tf.variable_scope('image_fc0'):
                W_fc0 = weight_variable(is_trainable=is_trainable,
                                        shape=[last_pool_image_dim * last_pool_image_dim * 512, 512],
                                        name='W_fc0')
                b_fc0 = bias_variable(is_trainable=is_trainable, shape=[512], name='b_fc0')
                if use_batch_norm:
                    self.h_fc0 = batch_norm_dense_activation(inputs=tf.nn.xw_plus_b(x=self.h_pool5_flat,
                                                                                    weights=W_fc0,
                                                                                    biases=b_fc0),
                                                             is_training=self.train_mode,
                                                             activation=activation,
                                                             is_trainable=is_trainable)
                else:
                    self.h_fc0 = activation(tf.matmul(self.h_pool5_flat, W_fc0) + b_fc0)
                self.h_fc0_drop = tf.nn.dropout(self.h_fc0, self.dropout_keep_prob)
            last_layer = self.h_fc0_drop
        else:
            last_layer = self.h_pool5_flat

        if is_trainable:
            # FC1 [512] -> [256]
            with tf.variable_scope('image_fc1'):
                W_fc1 = weight_variable(is_trainable=is_trainable, shape=[512, 256])
                b_fc1 = bias_variable(is_trainable=is_trainable, shape=[256])
                if use_batch_norm:
                    self.h_fc1 = batch_norm_dense_activation(inputs=tf.nn.xw_plus_b(x=last_layer,
                                                                                    weights=W_fc1,
                                                                                    biases=b_fc1),
                                                             is_training=self.train_mode,
                                                             activation=activation,
                                                             is_trainable=is_trainable)
                else:
                    self.h_fc1 = tf.nn.relu(tf.matmul(last_layer, W_fc1) + b_fc1)
                # dropout
                self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.dropout_keep_prob)

            # Softmax
            with tf.variable_scope('softmax'):
                self.W_softmax = weight_variable(is_trainable=is_trainable, shape=[256, NUM_CLASSES])
                self.b_softmax = bias_variable(is_trainable=is_trainable, shape=[NUM_CLASSES])
                self.probabilities = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_softmax) + self.b_softmax)

            # Finalize the predictions, the optimizing function, loss/accuracy stats etc.
            if self.is_primary_model:
                print("%s is a primary model, making optimizations" % self.model_name.name)
                self._set_predictions_optimizer_and_loss()
            else:
                print("%s not primary model, skipping optimizations" % self.model_name.name)

    def save_bottleneck_data(self, sess, layer_name, input_ids, input_labels, input_texts, output_filename):
        raise NotImplementedError


class ResnetClf(Resnet):
    def restore_base_models(self, sess):
        resnet_ckpt_filename = get_resnet_stored_filename(file_type='ckpt', num_layers=RESNET_LAYERS)
        # vars with name not matching current model and it's base text model will all be resnet vars
        vars_to_restore = \
            [v for v in tf.global_variables()
             if v.name.split('/')[0] != self.model_name.name]

        saver = tf.train.Saver(vars_to_restore)
        self.restore_model_from_filename(sess, model_filename=resnet_ckpt_filename, saver=saver)

    def __init__(self, is_trainable=True):
        self.model_name = ModelName.resnet_clf
        super(ResnetClf, self).__init__(model_name=self.model_name, is_trainable=True, train_last_block=False)

        # self.base_image_model = Resnet(restore=False, is_trainable=False, train_last_block=False)
        self.resnet_representation = self.avg_pool_representation

        activation = tf.nn.relu
        self.initializer_type = 'normal' if activation == tf.nn.relu else 'xavier'
        dim_D = 2048

        with tf.variable_scope(self.model_name.name):
            # FC1 [dim_d] -> [512]
            with tf.variable_scope('image_fc1'):
                W_fc1 = weight_variable(is_trainable=is_trainable, shape=[dim_D, 512],
                                        initializer_type=self.initializer_type, name='W_fc1')
                b_fc1 = bias_variable(is_trainable=is_trainable, shape=[512], name='b_fc1')

                if use_batch_norm:
                    h_fc1 = batch_norm_dense_activation(inputs=tf.matmul(self.resnet_representation, W_fc1) + b_fc1,
                                                        is_training=self.train_mode, activation=activation,
                                                        is_trainable=is_trainable)
                else:
                    h_fc1 = tf.nn.relu(tf.matmul(self.resnet_representation, W_fc1) + b_fc1)
                h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_keep_prob)

            # Softmax
            with tf.variable_scope('softmax'):
                W_softmax = weight_variable(is_trainable=is_trainable, shape=[512, NUM_CLASSES],
                                            initializer_type=self.initializer_type)
                b_softmax = bias_variable(is_trainable=is_trainable, shape=[NUM_CLASSES])
                self.scores = tf.matmul(h_fc1_drop, W_softmax) + b_softmax

            with tf.variable_scope('optimization'):
                # Finalize the predictions, the optimizing function, loss/accuracy stats etc.
                self._set_predictions_optimizer_and_loss()

    def save_bottleneck_data(self, sess, layer_name, input_ids, input_labels, input_texts, output_filename):
        raise NotImplementedError

    def _train_SGD_batch_step(self, sess, batch_image, batch_label, batch_ids, text_train, train_mode=True,
                              batch_step=1):
        # ignore batch_ids & text_train
        _, _ = batch_ids, text_train

        S, _, loss_val, acc_val = sess.run(
            [self.train_summary_op, self.train_op, self.loss, self.accuracy],
            feed_dict={
                # self.base_image_model.images_placeholder: batch_image,
                # self.base_image_model.train_mode: train_mode,
                self.images_placeholder: batch_image,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 0.5,
                self.train_mode: train_mode
            })

        return S, loss_val, acc_val

    def _validation_batch_step(self, sess, batch_image, batch_label, batch_ids, text_val, train_mode=False,
                               batch_step=1):
        # ignore batch_ids & text_train
        _ = batch_ids, text_val, batch_step

        val_acc_step = sess.run(
            self.sum_accuracy,
            feed_dict={
                self.images_placeholder: batch_image,
                # self.base_image_model.train_mode: train_mode,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 1.0,
                self.train_mode: train_mode
            })

        return val_acc_step

    def _test_batch_step(self, sess, c_mat, batch_image, batch_label, batch_ids, text_test, train_mode=False):
        # ignore batch_ids & text_test
        _, _ = batch_ids, text_test

        pred, mat, test_acc = sess.run(
            [self.probabilities, c_mat, self.sum_accuracy],
            feed_dict={
                self.images_placeholder: batch_image,
                # self.base_image_model.train_mode: train_mode,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 1.0,
                self.train_mode: train_mode
            })
        return pred, mat, test_acc
