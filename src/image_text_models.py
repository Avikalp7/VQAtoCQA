"""Module implementing image-text models for CQA-tasks.
"""
# Copyright (C) 2018 Yahoo Japan Corporation (Licensed under CC BY-NC-SA 4.0)
# https://creativecommons.org/licenses/by-nc-sa/4.0/


import os

# Elevating TF log level to remove tensorflow CPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# from tensorflow.contrib.layers import batch_norm
import numpy as np

from base_model import BaseModel
from image_models import Resnet
from text_models import TextCNN, HieText
from paths import get_stored_checkpoint_filename, data_directory, image_match_data_subdirectory, \
    get_resnet_stored_filename
from network import weight_variable, bias_variable, conv2d, batch_norm_conv_activation, batch_norm_dense_activation, \
    standard_FC_layer, get_initializer_type, standard_conv_layer
from utils import update_train_batch, update_val_batch, \
    update_test_batch, training_epoch_finish_routine, test_finish_routine, text_batch_from_ids
from global_hyperparams import NUM_CLASSES, ModelType, ModelName, best_date, best_epochs, \
    batch_size_dict, training_epochs_dict, NUM_TEXT_IN_MULTI_CHOICE, IMAGE_SIZE, scale_learning_rate, \
    decay_learning_rate, scale_factors_dict, scale_epochs_dict, NUM_IMAGES, RESNET_LAYERS, use_batch_norm, \
    early_stopping_learning_rate, USE_2014_DATA, TRAIN_RESNET_LAST_BLOCK, OPTIMIZER_MOMENTUM, \
    OptimizerType, OPTIMIZER, TEXT_LENTH, EMBED_SIZES

# Requires downloading the compacti bilinear pooling repository from
# https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling
try:
    from compact_bilinear_pooling import compact_bilinear_pooling_layer
except ImportError:
    compact_bilinear_pooling_layer = None


class ImageTextModel(BaseModel):
    """Base model class for all image-text based models

    * Implement a common train, val and test routine for all

    * Implement the common tensorboard visualization

    * Call the contructor the base image model, which is a Resnet
    """

    def __init__(self, model_name, is_trainable, config=None):
        """Call the BaseModel constructor, set common attributes

        Parameters
        ----------
        :param model_name: ModelName Enum from global_hyperparams

        :param config : dict, contains param name-value pairs for writing config file
        """
        # Call the BaseModel constructor
        super(ImageTextModel, self).__init__(model_name=model_name, model_type=ModelType.image_text, config=config,
                                             is_trainable=is_trainable)
        # Construct the base image Resnet model
        self.base_image_model = Resnet(restore=False, is_trainable=False, train_last_block=TRAIN_RESNET_LAST_BLOCK)
        # Define some common class attributes
        self.base_text_model = None
        self.base_text_model_name = None
        self.base_image_model_name = self.base_image_model_name = self.base_image_model.model_name
        self.base_image_model_last_layer = None
        self.v_I = None
        self.v_T = None
        self.activation = None
        self.initializer_type = None

    def restore_base_models(self, sess):
        """Restore pretrained image model that forms a component of the full model

        Raises
        ------
        ValueError: if self.config['base_image_model'] is neither of ModelName.image_cnn_v2 and ModelName.resnet
        """
        if self.config['base_image_model'] == ModelName.image_cnn_v2:
            image_model_ckpt_filename = get_stored_checkpoint_filename(
                model_type=self.base_image_model.model_type,
                model_name=self.base_image_model.model_name,
                date=best_date[self.base_image_model.model_name],
                num_epochs=best_epochs[self.base_image_model.model_name]
            )
            vars_to_restore = \
                [v for v in tf.global_variables()
                 if v.name.split('/')[0] != self.model_name.name
                 and v.name.split('/')[0] != self.base_text_model.model_name.name]
            saver = tf.train.Saver(vars_to_restore)
            self.base_image_model.restore_model_from_filename(sess=sess, model_filename=image_model_ckpt_filename,
                                                              saver=saver)
        elif self.config['base_image_model'] == ModelName.resnet:
            resnet_ckpt_filename = get_resnet_stored_filename(file_type='ckpt', num_layers=RESNET_LAYERS)
            # vars with name not matching current model and it's base text model will all be resnet vars
            vars_to_restore = \
                [v for v in tf.global_variables()
                 if v.name.split('/')[0] != self.model_name.name
                 and v.name.split('/')[0] != self.base_text_model.model_name.name]
            saver = tf.train.Saver(vars_to_restore)
            self.restore_model_from_filename(sess, model_filename=resnet_ckpt_filename, saver=saver)

        else:
            raise ValueError("Unimplemented base_image_model")

    def _train_helper_SGD(self, sess, train_writer, train_ids, train_labels, train_texts,
                          val_ids, val_labels, val_texts, iterator_train_init,
                          next_element_train, iterator_val_init, next_element_val):
        """Run mini-batch gradient descent on the training data

        :param sess: tf.Session() object

        :param train_writer: tf.summary.FileWriter object, file writer for TensorBoard

        :param train_ids: np.array of shape [NUM_TRAIN_SAMPLES] and type int, containing Chiebukuro question ids

        :param train_labels: np.array of shape [NUM_TRAIN_SAMPLES, NUM_CLASSES] of type bool,
            containing category labels

        :param train_texts: np.array of shape [NUM_TRAIN_SAMPLES, TEXT_LENGTH] of type int

        :param val_ids: np.array of shape [NUM_VAL_SAMPLES] and type int

        :param val_labels: np.array of shape [NUM_VAL_SAMPLES, NUM_CLASSES] and type bool

        :param val_texts: np.array of shape [NUM_VAL_SAMPLES, TEXT_LENGTH] and type int

        :param iterator_train_init: tf.contrib.data.TFRecordDataset.Iterator.initializeer object for train data

        :param next_element_train: tf operation, on run generates next batch (using iterator.get_next())

        :param iterator_val_init: tf.contrib.data.TFRecordDataset.Iterator.initializeer object for validation data

        :param next_element_val: tf operation, on run generates next batch (using iterator.get_next())
        """
        text_train, text_val = train_texts, val_texts

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
                                               train_labels, text_train, train_writer, global_step)

            # Now performing validation
            num_val_samples, val_acc_sum = self._val_prediction_loop(sess, iterator_val_init,
                                                                     next_element_val, val_labels,
                                                                     text_val, epoch)

            training_epoch_finish_routine(sess, val_acc_sum, num_val_samples, self.train_logfile_name,
                                          self.checkpoint_dir, epoch, self.saver)
            val_acc_history.append(val_acc_sum / float(num_val_samples))
            print ('Previous Val Accs:', val_acc_history[-10:])
            epochs_with_current_lrate += 1

        # Completed all epochs
        print('BEST VAL: ', max(val_acc_history))
        # noinspection PyUnboundLocalVariable
        print("Completed training %s model, with %d epochs" % (self.model_name.name, epoch + 1))

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
        text_test = test_texts

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
        total_samples = 0
        all_batch_ids = []
        while True:
            try:
                batch_image, _, batch_ids = sess.run(next_element_test)
                all_batch_ids += list(batch_ids)
                total_samples += len(batch_ids)
            except tf.errors.OutOfRangeError:
                # Completed all minibatches in test set
                break

            pred, mat, test_acc = self._test_batch_step(sess, c_mat,
                                                        batch_image, test_labels[batch_ids], batch_ids, text_test)

            conf_matrix, prediction, test_accuracy = \
                update_test_batch(batch_num, pred, mat, test_acc, conf_matrix, prediction, test_accuracy, total_batches)
            batch_num += 1

        labels, results = test_finish_routine(test_labels[all_batch_ids], prediction)
        test_accuracy /= float(total_samples)

        return test_accuracy, labels, results, conf_matrix

    def _run_validation_test(self, sess, val_ids, val_labels, val_texts, iterator_validation_init, next_element_val):
        text_val = val_texts
        val_acc_sum, num_samples = self._val_prediction_loop(sess, iterator_validation_init, next_element_val,
                                                             val_labels, text_val, 0)
        print("*** Validation accuracy: %0.4f ***" % (val_acc_sum / float(num_samples)))

    def _train_SGD_batch_step(self, sess, batch_image, batch_label, batch_ids, text_train, batch_step=None):
        """ Abstract method """
        raise NotImplementedError

    def _validation_batch_step(self, sess, batch_image, batch_label, batch_ids, text_val, batch_step=None):
        """ Abstact method """
        raise NotImplementedError

    def _test_batch_step(self, sess, c_mat, batch_image, batch_label, batch_ids, text_test):
        """ Abstract method """
        raise NotImplementedError

    def _get_derived_image_representation_dimensions(self):
        """Originally there was a choice between a from-scratch-image-cnn and ResNet,
            leading to different derived dims

        Returns
        -------
        image_dim: int, m if the spatial image representation is m*m

        image_depth:int,  dimension for the embedding of each region
        """
        image_division_factor = 16 if self.base_image_model.model_name == ModelName.image_cnn_v2 else 32
        image_dim = int(IMAGE_SIZE / image_division_factor)
        image_depth = 512 if self.base_image_model.model_name == ModelName.image_cnn_v2 else 2048
        return image_dim, image_depth

    def _dim_reduction_with_one_one_conv(self, image_dim, dim_D, dim_k, is_trainable):
        """Reduce a 4D tensor's last dimension using 1x1 convolutions

        Parameters
        ----------
        :param image_dim: int, m if the spatial image representation is m*m

        :param dim_D: int, last dimension

        :param dim_k: int, dimension to which to reduce to

        :param is_trainable: whether the convolution weights will be trainable

        Returns
        -------
        conv_output: 4-d tensor, output of applying the convolution
        """
        f_I_spatial = tf.reshape(self.base_image_model_last_layer,
                                 shape=[-1, image_dim, image_dim, dim_D],
                                 name='f_I_spatial')
        # [filter_size, filter_size, channel_size, num_filters]
        W_conv = weight_variable(is_trainable=is_trainable, shape=[1, 1, dim_D, dim_k],
                                 initializer_type=self.initializer_type)
        b_conv = bias_variable(is_trainable=is_trainable, shape=[dim_k])
        if use_batch_norm:
            conv_output = batch_norm_conv_activation(is_trainable=is_trainable,
                                                     inputs=conv2d(x=f_I_spatial, W=W_conv) + b_conv,
                                                     is_training=self.train_mode, activation=self.activation)
        else:
            conv_output = self.activation(conv2d(x=f_I_spatial, W=W_conv) + b_conv)
        return conv_output


class EmbeddingConcatWithSemiFreeze(ImageTextModel):
    """Class implementing the Embedding-Concat, Sum-Prod-Concat & MCB models from the paper"""

    def __init__(self, is_trainable=True, dim_k=2048):
        """Implement the model components

        Parameters
        ----------
        :param is_trainable: bool, whether the model weights will be trainable

        :param dim_k: int, the common dimension to which derived image & text embeddings are projected
        """

        self.model_name = ModelName.embedding_concat_semifreeze
        # Set use_simple_concat to True to implement Embedding-Concat
        # Set use_add_mul_concat to True to implement Add-Mul-Concat
        # Set use_MCB to True to use non attention-based MCB
        self.config = {
            'TRAIN_RESNET_LAST_BLOCK': TRAIN_RESNET_LAST_BLOCK,
            'activation': tf.nn.tanh,
            'base_text_model': ModelName.text_cnn,
            'base_image_model': ModelName.resnet,
            'base_text_activation': tf.nn.relu,
            'base_text_filter_sizes': (1, 2, 3),
            'base_text_num_filters': (128, 256, 256),
            'use_conv_for_img_reduction': True,
            'use_add_mul_concat': True,
            'use_MCB': False,
            'use_simple_concat': False,
        }
        self.activation = self.config['activation']
        self.initializer_type = get_initializer_type(self.activation)
        self.base_text_model_name = self.config['base_text_model']
        self.base_image_model_last_layer = self.base_image_model.representation

        # Ensure that compact_bilinear_pooling module is present. See the github link at top
        if self.config['use_MCB'] and compact_bilinear_pooling_layer is None:
            raise ValueError("compact_bilinear_pooling_layer module not found. Can't implement MCB")

        # Ensure that configuration is properly set to just one of the models
        assert (int(self.config['use_add_mul_concat']) + int(self.config['use_MCB']) +
                int(self.config['use_simple_concat']) == 1)

        with tf.variable_scope(self.model_name.name):
            super(EmbeddingConcatWithSemiFreeze, self).__init__(model_name=self.model_name, config=self.config,
                                                                is_trainable=is_trainable)
            if self.config['base_text_model'] == ModelName.text_cnn:
                self.base_text_model = TextCNN(filter_sizes=self.config['base_text_filter_sizes'],
                                               num_filters=self.config['base_text_num_filters'],
                                               activation=self.config['base_text_activation'],
                                               is_trainable=is_trainable,
                                               is_primary_model=False)
            else:
                raise ValueError("No other text model yet supported for %s model" % self.model_name.name)

            # Deciding what dimensional image embedding we'd get depending on the base image model
            image_dim, image_depth = self._get_derived_image_representation_dimensions()
            dim_d = self.base_text_model.final_embedding_dimension
            dim_m = image_dim * image_dim
            dim_D = image_depth
            print("For EmbedConcat: dim_d = %d, dim_m = %d, dim_D = %d" % (dim_d, dim_m, dim_D))

            # If common dimension is different from derived image's dimension,
            # Apply transformation to bring it to dim_k
            if dim_D != dim_k:
                with tf.variable_scope('dimD_to_dimk'):
                    self.v_spI = self._dim_reduction_with_one_one_conv(image_dim, dim_D, dim_k, is_trainable)
                    v_spI_unfolded = tf.reshape(self.v_spI, shape=[-1, dim_m, dim_k])
                    self.v_I = tf.reduce_mean(v_spI_unfolded, axis=1)
            # Else if derived image dimension is same as common dimension, nothing to do
            else:
                self.v_spI = self.base_image_model_last_layer

            # Derive the flat embedding from the spatial representation
            v_spI_unfolded = tf.reshape(self.v_spI, shape=[-1, dim_m, dim_k], name='v_spI_unfolded')
            self.v_I = tf.reduce_mean(v_spI_unfolded, axis=1)

            if self.config['use_dropout_on_init_embeddings']:
                self.v_I = tf.nn.dropout(self.v_I, self.dropout_keep_prob)

            with tf.variable_scope("Init_Ques_Emb"):
                # v_T -> [B, dim_d]
                self.v_T = tf.reshape(self.base_text_model.h_pool, shape=[-1, dim_d])
                self.v_T = tf.nn.dropout(self.v_T, self.dropout_keep_prob)

            # If common dimension is different from derived text's dimension,
            # Apply transformation to bring it to dim_k
            if dim_d != dim_k:
                with tf.variable_scope('FCLayer_TextDim_to_dimk'):
                    self.v_T = standard_FC_layer(self.v_T, dim_d, dim_k, use_batch_norm, self.activation,
                                                 self.train_mode, is_trainable, self.dropout_keep_prob, 'text')

            with tf.variable_scope("Concat"):
                if self.config['use_MCB']:
                    text_embed = tf.reshape(self.v_T, shape=[-1, 1, 1, dim_k])
                    image_embed = tf.reshape(self.v_I, shape=[-1, 1, 1, dim_k])
                    embed_dim_big = dim_k * 4
                    embed_dim = int(dim_k / 2)
                    # noinspection PyCallingNonCallable
                    mcb_out = compact_bilinear_pooling_layer(bottom1=text_embed,
                                                             bottom2=image_embed,
                                                             output_dim=embed_dim_big)
                    with tf.variable_scope("MCB_FC"):
                        self.embed_concat = standard_FC_layer(mcb_out, embed_dim_big, embed_dim, use_batch_norm,
                                                              self.activation, self.train_mode, is_trainable,
                                                              self.dropout_keep_prob, name_suffix='mcb_fc')
                elif self.config['use_simple_concat']:
                    self.embed_concat = tf.concat((self.v_I, self.v_T), axis=1)
                    embed_dim = dim_k * 2
                else:
                    self.embed_add = self.v_I + self.v_T
                    self.embed_mul = self.v_I * self.v_T
                    self.embed_concat = tf.concat((self.embed_add, self.embed_mul), axis=1)
                    embed_dim = dim_k * 2

            with tf.variable_scope("softmax"):
                W_u = weight_variable(is_trainable=is_trainable, shape=[embed_dim, NUM_CLASSES], name='W_u')
                b_u = bias_variable(is_trainable=is_trainable, shape=[NUM_CLASSES], name='b_u')
                self.scores = tf.matmul(self.embed_concat, W_u) + b_u

            with tf.variable_scope("optimization"):
                # Finalize the predictions, the optimizing function, loss/accuracy stats etc.
                self._set_predictions_optimizer_and_loss()

    def _train_SGD_batch_step(self, sess, batch_image, batch_label, batch_ids, text_train, train_mode=True,
                              batch_step=1):
        """Run the TF's graph to perform GD over the current batch"""
        S, _, loss_val, acc_val = sess.run(
            [self.train_summary_op, self.train_op, self.loss, self.accuracy],
            feed_dict={
                self.train_mode: train_mode,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 0.5,
                self.base_image_model.images_placeholder: batch_image,
                self.base_image_model.train_mode: train_mode,
                self.base_text_model.texts_placeholder: text_batch_from_ids(batch_ids, text_data=text_train),
                self.base_text_model.embedding_dropout: 0.3,
                self.base_text_model.embedding_noise_std: 0.01,
                self.base_text_model.train_mode: train_mode,
                self.base_text_model.phase_train: train_mode
            })

        return S, loss_val, acc_val

    def _validation_batch_step(self, sess, batch_image, batch_label, batch_ids, text_val, train_mode=False,
                               batch_step=1):
        val_acc = sess.run(
            self.sum_accuracy,
            feed_dict={
                self.train_mode: train_mode,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 1.,
                self.base_image_model.images_placeholder: batch_image,
                self.base_image_model.train_mode: train_mode,
                self.base_text_model.texts_placeholder: text_batch_from_ids(batch_ids, text_data=text_val),
                self.base_text_model.embedding_dropout: 0.,
                self.base_text_model.embedding_noise_std: 0.,
                self.base_text_model.train_mode: train_mode,
                self.base_text_model.phase_train: train_mode
            })

        return val_acc

    def _test_batch_step(self, sess, c_mat, batch_image, batch_label, batch_ids, text_test,
                         train_mode=False):
        pred, mat, test_acc = sess.run(
            [self.probabilities, c_mat, self.sum_accuracy],
            feed_dict={
                self.train_mode: train_mode,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 1.,
                self.base_image_model.images_placeholder: batch_image,
                self.base_image_model.train_mode: train_mode,
                self.base_text_model.texts_placeholder: text_batch_from_ids(batch_ids, text_data=text_test),
                self.base_text_model.embedding_dropout: 0.,
                self.base_text_model.embedding_noise_std: 0.,
                self.base_text_model.train_mode: train_mode,
                self.base_text_model.phase_train: train_mode
            })

        return pred, mat, test_acc


class StackedAttentionWithSemiFreezeCNN(ImageTextModel):
    def __init__(self, nlayers=1, dim_k=1024, dim_att=512, is_trainable=True):
        """Contruct the SAN model, with options for model w/ and w/o global image weight

        * Implement the SAN layer-1 and layer-2 according to passed argument

        * Implement the global image weight model according to config

        Parameters
        ----------
        :param nlayers: int, number of stacked layers in the network

        :param dim_k: int, the common dimension for image and text representations

        :param dim_att: int, the dimensions of the attention layer representation

        :param is_trainable: bool, whether weights of the model can be updated
        """
        self.model_name = ModelName.stacked_attention_with_semi_freeze_cnn
        self.config = {
            'TRAIN_RESNET_LAST_BLOCK': TRAIN_RESNET_LAST_BLOCK,
            'activation': tf.nn.tanh,
            'base_text_model': ModelName.text_cnn,
            'base_image_model': ModelName.resnet,
            'base_text_activation': tf.nn.relu,
            'base_text_filter_sizes': (1, 2, 3),
            'base_text_num_filters': (128, 256, 256),
            'num_layers': nlayers,
            'dim_k': dim_att,
            'include_global_image_wt': False,
            'use_prod_in_embed': True,
        }
        self.activation = self.config['activation']
        self.initializer_type = get_initializer_type(self.activation)
        self.base_text_model_name = self.config['base_text_model']
        self.base_image_model_last_layer = self.base_image_model.representation

        with tf.variable_scope(self.model_name.name):
            super(StackedAttentionWithSemiFreezeCNN, self).__init__(model_name=self.model_name,
                                                                    config=self.config,
                                                                    is_trainable=is_trainable)

            # Construct the base text model
            if self.config['base_text_model'] == ModelName.text_cnn:
                self.base_text_model = TextCNN(filter_sizes=self.config['base_text_filter_sizes'],
                                               num_filters=self.config['base_text_num_filters'],
                                               activation=self.config['base_text_activation'],
                                               is_trainable=is_trainable,
                                               is_primary_model=False)
            else:
                raise ValueError("No other text model yet supported for %s model" % self.model_name.name)

            # Deciding what dimensional image embedding we'd get depending on the base image model
            image_dim, image_depth = self._get_derived_image_representation_dimensions()
            text_embedding_dimension = sum(self.config['base_text_num_filters'])
            dim_d = text_embedding_dimension
            dim_m = image_dim * image_dim
            dim_D = image_depth
            print("For Stacked CNN: dim_d = %d, dim_m = %d, dim_D = %d" % (dim_d, dim_m, dim_D))

            # Convert f_I of dimensions [B*dim_m, dim_D] to v_I of dimensions [B*dim_m, dim_d]
            with tf.variable_scope('dimD_to_dimk'):
                self.v_spI = self._dim_reduction_with_one_one_conv(image_dim, dim_D, dim_k, is_trainable)
                v_spI_unfolded = tf.reshape(self.v_spI, shape=[-1, dim_m, dim_k])
                self.v_I = tf.reduce_mean(v_spI_unfolded, axis=1)

            with tf.variable_scope("Init_Ques_Emb"):
                # v_Q -> [B, dim_d]
                self.v_T = tf.reshape(self.base_text_model.h_pool,
                                      shape=[-1, self.base_text_model.final_embedding_dimension])
                self.v_T = tf.nn.dropout(self.v_T, self.dropout_keep_prob)
                self.u = [None] * (nlayers + 1)
                self.u[0] = self.v_T

            if dim_d != dim_k:
                with tf.variable_scope('FCLayer_TextDim_to_dimk'):
                    self.u[0] = standard_FC_layer(self.u[0], dim_d, dim_k,
                                                  use_batch_norm, self.activation, self.train_mode,
                                                  is_trainable, self.dropout_keep_prob, 'text')

            uprod = None
            with tf.variable_scope("stacked_attention"):
                for layer_num in range(nlayers):
                    self.u[layer_num + 1], uprod = self._stack_attention_layer(v_Q=self.u[layer_num],
                                                                               layer_num=(layer_num + 1),
                                                                               v_spI=self.v_spI, dim_k=dim_k,
                                                                               dim_m=dim_m, dim_att=dim_att,
                                                                               is_trainable=is_trainable)

            with tf.variable_scope("final_embedding"):
                if self.config['use_prod_in_embed']:
                    self.final_embedding = tf.concat((self.u[nlayers], uprod), axis=1, name='final_embedding')
                else:
                    self.final_embedding = self.u[nlayers]

            with tf.variable_scope("softmax"):
                W_u = weight_variable(is_trainable=is_trainable,
                                      shape=[self.final_embedding.shape[1].value, NUM_CLASSES],
                                      name='W_u')
                b_u = bias_variable(is_trainable=is_trainable, shape=[NUM_CLASSES], name='b_u')
                self.scores = tf.matmul(self.final_embedding, W_u) + b_u

            with tf.variable_scope("optimization"):
                # Finalize the predictions, the optimizing function, loss/accuracy stats etc.
                self._set_predictions_optimizer_and_loss()

    def _stack_attention_layer(self, v_Q, layer_num, v_spI, dim_k, dim_m, dim_att, is_trainable):
        """Implement the attention layer in SAN

        Parameters
        ----------
        :param v_Q: tensor, derived text embedding of shape [B, dim_k]

        :param layer_num: int, the 0-indexed layer number

        :param v_spI: tensor, derived spatial image representation of shape [B*dim_m, dim_k]

        :param dim_k: int, the dimension of derived image and text representations

        :param dim_m: int, the number of regions in the image

        :param dim_att: int, the attention layer dimension

        :param is_trainable: bool, whether the weights are trainable

        Returns
        -------
        u_sum: Vt + Vi, of shape [B, dim_k]
        u_prod: Vt * Vi, of shape [B, dim_k]
        """
        with tf.variable_scope('FCLayer_Attention_%d' % layer_num):
            W_IA = weight_variable(is_trainable=is_trainable, shape=[dim_k, dim_att],
                                   initializer_type=self.initializer_type, name='W_IA')
            W_QA = weight_variable(is_trainable=is_trainable, shape=[dim_k, dim_att],
                                   initializer_type=self.initializer_type, name='W_QA')
            b_A = bias_variable(is_trainable=is_trainable, shape=[dim_att], name='b_A')

            # question_prod -> [B, dim_att]
            question_prod = tf.nn.xw_plus_b(x=v_Q, weights=W_QA, biases=b_A, name='question_prod_%d' % layer_num)
            # question_prod -> [B, 1, dim_att]
            question_prod = tf.expand_dims(input=question_prod, axis=1)

            v_spI_dash = None
            if self.config['include_global_image_wt']:
                with tf.variable_scope('global_image_wt_%d' % layer_num):
                    W_fb = weight_variable(is_trainable=is_trainable, shape=[dim_k, dim_k],
                                           initializer_type=self.initializer_type, name='W_fb')
                    b_fb = bias_variable(is_trainable=is_trainable, shape=[dim_k], name='b_fb')
                    # text_feature shape -> [B, dim_k]
                    v_T_dash = tf.nn.xw_plus_b(x=v_Q, weights=W_fb, biases=b_fb, name='text_feature')

                    if use_batch_norm:
                        v_T_dash = batch_norm_dense_activation(v_T_dash,
                                                               is_training=self.train_mode,
                                                               activation=self.activation,
                                                               is_trainable=is_trainable)
                    else:
                        v_T_dash = self.activation(v_T_dash)

                    v_T_dash = tf.nn.dropout(v_T_dash, self.dropout_keep_prob)
                    v_T_dash_unrolled = tf.expand_dims(v_T_dash, axis=1)
                    v_spI_unrolled = tf.reshape(v_spI, shape=[-1, dim_m, dim_k])
                    # [B, dim_m + 1, dim_k]
                    v_spI_dash_unrolled = tf.concat((v_spI_unrolled, v_T_dash_unrolled), axis=1)
                    # [B * (dim_m + 1), dim_k]
                    v_spI_dash = tf.reshape(v_spI_dash_unrolled, shape=[-1, dim_k])
                    # [B * (dim_m + 1), dim_att]
                    image_prod = tf.matmul(v_spI_dash, W_IA)
                    # image_prod_unrolled -> [B, dim_m+1, dim_att]
                    image_prod_unrolled = tf.reshape(image_prod, shape=[-1, dim_m+1, dim_k])
                    new_dim_m = dim_m + 1
            else:
                # image_prod -> [B*dim_m, dim_att]
                image_prod = tf.matmul(v_spI, W_IA)
                # image_prod_unrolled -> [B, dim_m, dim_att]
                image_prod_unrolled = tf.reshape(image_prod, shape=[-1, dim_m, dim_k])
                new_dim_m = dim_m

            # image_question_prod_sum -> [B*dim_m(+1), dim_k]
            image_question_prod_sum = tf.reshape(image_prod_unrolled + question_prod, shape=[-1, dim_att])

            # h_A -> [B*dim_m(+1), dim_k]
            if use_batch_norm:
                h_A = batch_norm_dense_activation(inputs=image_question_prod_sum, activation=self.activation,
                                                  is_training=self.train_mode, is_trainable=is_trainable)
            else:
                h_A = self.activation(image_question_prod_sum)
            h_A_drop = tf.nn.dropout(h_A, self.dropout_keep_prob)
            # h_A_drop_folded -> [B*dim_m(+1), dim_att]
            h_A_drop_folded = h_A_drop

        with tf.variable_scope('Softmax_Attention_%d' % layer_num):
            W_P = weight_variable(is_trainable=is_trainable, shape=[dim_att, 1], name='W_P')
            b_P = bias_variable(is_trainable=is_trainable, shape=[1])
            p_I_unfolded = tf.nn.softmax(tf.reshape(tf.nn.xw_plus_b(x=h_A_drop_folded,
                                                                    weights=W_P,
                                                                    biases=b_P),
                                                    shape=[-1, new_dim_m]))
            # p_I -> [B*dim_m(+1), 1]
            p_I = tf.reshape(p_I_unfolded, shape=[-1, 1])

        with tf.variable_scope('Attention_Weighting_%d' % layer_num):
            # v_I_weighted -> [B*dim_m(+1), dim_k]
            if self.config['include_global_image_wt']:
                v_I_weighted = v_spI_dash * p_I
            else:
                v_I_weighted = v_spI * p_I

            v_I_weighted = tf.reshape(v_I_weighted,
                                      shape=[-1, new_dim_m, dim_k],
                                      name='v_I_weighted_%d' % layer_num)
            # v_I_cap -> [B, dim_k]
            v_I_cap = tf.reduce_sum(v_I_weighted, axis=1, name='v_I_cap_%d' % layer_num)

        with tf.variable_scope('New_Query_%d' % layer_num):
            # u -> [B, dim_k]
            u_sum = v_I_cap + v_Q
            u_prod = v_I_cap * v_Q

        return u_sum, u_prod

    def _train_SGD_batch_step(self, sess, batch_image, batch_label, batch_ids, text_train, train_mode=True,
                              batch_step=1):
        S, _, loss_val, acc_val = sess.run(
            [self.train_summary_op, self.train_op, self.loss, self.accuracy],
            feed_dict={
                self.train_mode: train_mode,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 0.5,
                self.base_image_model.images_placeholder: batch_image,
                self.base_image_model.train_mode: False,
                self.base_text_model.texts_placeholder: text_batch_from_ids(batch_ids, text_data=text_train),
                self.base_text_model.embedding_dropout: 0.3,
                self.base_text_model.embedding_noise_std: 0.01,
                self.base_text_model.train_mode: train_mode,
                self.base_text_model.phase_train: train_mode,
            })

        return S, loss_val, acc_val

    def _validation_batch_step(self, sess, batch_image, batch_label, batch_ids, text_val, train_mode=False,
                               batch_step=1):
        val_acc = sess.run(
            self.sum_accuracy,
            feed_dict={
                self.train_mode: train_mode,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 1.,
                self.base_image_model.images_placeholder: batch_image,
                self.base_image_model.train_mode: True,
                self.base_text_model.texts_placeholder: text_batch_from_ids(batch_ids, text_data=text_val),
                self.base_text_model.embedding_dropout: 0.,
                self.base_text_model.embedding_noise_std: 0.,
                self.base_text_model.train_mode: train_mode,
                self.base_text_model.phase_train: train_mode,
            })

        return val_acc

    def _test_batch_step(self, sess, c_mat, batch_image, batch_label, batch_ids, text_test,
                         train_mode=False):
        pred, mat, test_acc = sess.run(
            [self.probabilities, c_mat, self.sum_accuracy],
            feed_dict={
                self.train_mode: train_mode,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 1.,
                self.base_image_model.images_placeholder: batch_image,
                self.base_image_model.train_mode: True,
                self.base_text_model.texts_placeholder: text_batch_from_ids(batch_ids, text_data=text_test),
                self.base_text_model.embedding_dropout: 0.,
                self.base_text_model.embedding_noise_std: 0.,
                self.base_text_model.train_mode: train_mode,
                self.base_text_model.phase_train: train_mode,
            })

        return pred, mat, test_acc


class AuxTaskModel(ImageTextModel):
    def __init__(self, dim_k=1024, dim_att=256, is_trainable=True, is_primary_model=True):
        self.model_name = ModelName.aux_task_model
        self.config = {
            'TRAIN_RESNET_LAST_BLOCK': TRAIN_RESNET_LAST_BLOCK,
            'activation': tf.nn.tanh,
            'base_text_model': ModelName.text_cnn,
            'base_image_model': ModelName.resnet,
            'base_text_activation': tf.nn.relu,
            'base_text_filter_sizes': (1, 2, 3),
            'base_text_num_filters': (128, 256, 256),
            'embedding_size_multiplier': 1.5,
            'dim_k': dim_k,
            'dim_att': dim_att,
            'use_prod_in_embed': True,
        }
        self.activation = self.config['activation']
        self.initializer_type = get_initializer_type(self.activation)
        self.base_text_model_name = self.config['base_text_model']
        self.base_image_model_last_layer = self.base_image_model.representation

        # Load the train and validation data if in train mode, and test data if in test mode
        self._load_aux_data()

        with tf.variable_scope(self.model_name.name):
            super(AuxTaskModel, self).__init__(model_name=self.model_name, config=self.config,
                                               is_trainable=is_trainable)

            # Define model specific placeholders
            self.aux_labels_placeholder = tf.placeholder(dtype="float", shape=[None, NUM_TEXT_IN_MULTI_CHOICE],
                                                         name='aux_labels_placeholder')
            # Whether the task is image-to-texts mathcing, or text-to-images matching
            self.image_to_texts_bool_placeholder = tf.placeholder(dtype=tf.bool, name='image_to_texts_bool_placeholder')

            self.aux_probabilities = None

            if self.config['base_text_model'] == ModelName.text_cnn:
                self.base_text_model = TextCNN(filter_sizes=self.config['base_text_filter_sizes'],
                                               num_filters=self.config['base_text_num_filters'],
                                               activation=self.config['base_text_activation'],
                                               is_trainable=is_trainable,
                                               is_primary_model=False,
                                               embed_size_multiplier=self.config['embedding_size_multiplier'])
            else:
                raise ValueError("No other text model yet supported for %s model" % self.model_name.name)

            # Deciding what dimensional image embedding we'd get depending on the base image model
            image_dim, image_depth = self._get_derived_image_representation_dimensions()
            text_embedding_dimension = sum(self.config['base_text_num_filters'])
            dim_d = text_embedding_dimension
            dim_m = image_dim * image_dim
            dim_D = image_depth
            print("For Stacked CNN: dim_d = %d, dim_m = %d, dim_D = %d" % (dim_d, dim_m, dim_D))

            with tf.variable_scope('dimD_to_dimk'):
                self.v_spI = self._dim_reduction_with_one_one_conv(image_dim, dim_D, dim_k, is_trainable)
                v_spI_unfolded = tf.reshape(self.v_spI, shape=[-1, dim_m, dim_k])
                self.v_I = tf.reduce_mean(v_spI_unfolded, axis=1)

            with tf.variable_scope("Init_Ques_Emb"):
                # v_T -> [B(*N), dim_d]
                self.v_T = tf.reshape(self.base_text_model.h_pool,
                                      shape=[-1, self.base_text_model.final_embedding_dimension])
                self.v_T = tf.nn.dropout(self.v_T, self.dropout_keep_prob, name='v_T')

            if dim_d != dim_k:
                with tf.variable_scope('FCLayer_TextDim_to_dimk'):
                    self.v_T = standard_FC_layer(self.v_T, dim_d, dim_k, use_batch_norm, self.activation,
                                                 self.train_mode, is_trainable, self.dropout_keep_prob, 'text')

            with tf.variable_scope("add_mul_concat"):
                # If the aux task is image-to-texts, we tile the image
                if self.image_to_texts_bool_placeholder:
                    # v_I_expanded -> [B(*N), 1, dim_k]
                    v_I_expanded = tf.expand_dims(self.v_I, axis=1)
                    # v_T_unfoled -> [B, NUM_TEXT_IN_MULTI_CHOICE, dim_k]
                    v_T_unfolded = tf.reshape(self.v_T, shape=[-1, NUM_TEXT_IN_MULTI_CHOICE, dim_k])
                    self.embedding_add = v_I_expanded + v_T_unfolded
                    self.embedding_mul = v_I_expanded * v_T_unfolded
                # Else if the aux task is text-to-images, we tile the text
                else:
                    # v_T_expanded -> [B(*N), 1, dim_k]
                    v_T_expanded = tf.expand_dims(self.v_T, axis=1)
                    # v_I_unfolded -> [B, NUM_TEXT_IN_MULTI_CHOICE, dim_k]
                    v_I_unfolded = tf.reshape(self.v_I, shape=[-1, NUM_TEXT_IN_MULTI_CHOICE, dim_k])
                    self.embedding_add = v_T_expanded + v_I_unfolded
                    self.embedding_mul = v_T_expanded * v_I_unfolded

                # embedding_concat -> [B, NUM_TEXT_IN_MULTI_CHOICE, 2*tdim]
                self.embedding_concat = tf.concat(values=(self.embedding_add, self.embedding_mul),
                                                  axis=2,
                                                  name="embedding_concat")
                # embedding_concat_4dtensor -> [B, NUM_TEXT_IN_MULTI_CHOICE, 1, 2*tdim]
                self.embedding_concat_4dtensor = tf.reshape(self.embedding_concat,
                                                            shape=[-1, NUM_TEXT_IN_MULTI_CHOICE, 1, 2 * dim_k],
                                                            name="embedding_concat_4dtensor")
                self.final_embedding = self.embedding_concat_4dtensor

                with tf.variable_scope("conv_layer_1"):
                    filter_shape = [1, 1, 2 * dim_k if self.config['use_prod_in_embed'] else dim_k, 256]
                    h_conv1 = standard_conv_layer(self.final_embedding, filter_shape, self.activation,
                                                  use_batch_norm, self.train_mode, is_trainable, 'conv1')

                with tf.variable_scope("conv_layer_2"):
                    filter_shape = [1, 1, 256, 1]
                    h_conv2 = standard_conv_layer(h_conv1, filter_shape, self.activation,
                                                  use_batch_norm, self.train_mode, is_trainable, 'conv2')

                with tf.variable_scope("conv_softmax_output"):
                    h_conv_out = tf.reshape(h_conv2, shape=[-1, NUM_TEXT_IN_MULTI_CHOICE], name='h_conv_out')
                    self.aux_probabilities = tf.nn.softmax(h_conv_out, name="aux_probabilities")

            if is_primary_model:
                with tf.variable_scope("optimization"):
                    self._set_aux_predictions_optimizer_and_loss()

    def _load_aux_data(self):
        if self.trainable:
            self.aux_train_labels = \
                np.load(data_directory + image_match_data_subdirectory + 'train_new_labels%d%s.npy'
                        % (NUM_CLASSES, '_2014' if USE_2014_DATA else ''))
            self.aux_text_train_ids = \
                np.load(data_directory + image_match_data_subdirectory + 'train_text_ids%d%s.npy'
                        % (NUM_CLASSES, '_2014' if USE_2014_DATA else ''))
            self.aux_val_labels = \
                np.load(data_directory + image_match_data_subdirectory + 'val_new_labels%d%s.npy'
                        % (NUM_CLASSES, '_2014' if USE_2014_DATA else ''))
            self.aux_text_val_ids = \
                np.load(data_directory + image_match_data_subdirectory + 'val_text_ids%d%s.npy'
                        % (NUM_CLASSES, '_2014' if USE_2014_DATA else ''))
        else:
            self.aux_train_labels, self.aux_text_train_ids, self.aux_val_labels, self.aux_text_val_ids = \
                None, None, None, None
            self.aux_test_labels = \
                np.load(data_directory + image_match_data_subdirectory + 'test_new_labels%d%s.npy'
                        % (NUM_CLASSES, '_2014' if USE_2014_DATA else ''))
            self.aux_text_test_ids = \
                np.load(data_directory + image_match_data_subdirectory + 'test_text_ids%d%s.npy'
                        % (NUM_CLASSES, '_2014' if USE_2014_DATA else ''))

    def _set_aux_predictions_optimizer_and_loss(self):
        self.aux_predictions = tf.argmax(self.aux_probabilities, 1, name="aux_predictions")

        if self.trainable:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            self.aux_loss = -tf.reduce_sum(
                self.aux_labels_placeholder * tf.log(tf.clip_by_value(self.aux_probabilities, 1e-10, 1.0)),
                name='aux_loss')
            self.loss = tf.constant(0., dtype="float", name='dummy_loss')
            self.total_loss = self.aux_loss

        if decay_learning_rate:
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                            3000, 0.90, staircase=True)
        elif scale_learning_rate:
            initial_learning_rate = self.learning_rate * batch_size_dict[self.model_type] / 256.
            batches_per_epoch = int(NUM_IMAGES['train'] / batch_size_dict[self.model_type])
            # Multiply the learning rate by 0.5 at 2, 4, 80, and 90 epochs.
            boundaries = [
                int(batches_per_epoch * epoch) for epoch in scale_epochs_dict[self.model_type]]
            values = [
                initial_learning_rate * scale for scale in scale_factors_dict[self.model_type]]
            self.learning_rate = tf.train.piecewise_constant(
                tf.cast(self.global_step, tf.int32), boundaries, values)

        elif early_stopping_learning_rate:
            self.learning_rate = tf.Variable(self.learning_rate, trainable=False, name='learning_rate_var')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_name.name)

        if self.base_image_model.model_name == ModelName.image_cnn_v2 and self.config['train_image_last_layers']:
            imgcnn_update_ops1 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='image_conv_8')
            imgcnn_update_ops2 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='image_conv_7')
            imgcnn_update_ops3 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='image_conv_6')
            update_ops = update_ops + imgcnn_update_ops1 + imgcnn_update_ops2 + imgcnn_update_ops3

        with tf.control_dependencies(update_ops):
            # Define Training procedure
            if OPTIMIZER == OptimizerType.adam:
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif OPTIMIZER == OptimizerType.rms_optimizer:
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                           momentum=OPTIMIZER_MOMENTUM)
            else:
                raise ValueError("Invalid value for OPTIMIZER var")

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

        # Calculate Accuracy
        with tf.variable_scope("accuracy"):
            self.aux_correct_predictions = tf.equal(self.aux_predictions, tf.argmax(self.aux_labels_placeholder, 1))
            self.aux_accuracy = tf.reduce_mean(tf.cast(self.aux_correct_predictions, "float"), name="aux_accuracy")
            self.aux_sum_accuracy = tf.reduce_sum(tf.cast(self.aux_correct_predictions, "float"),
                                                  name="aux_sum_accuracy")
            self.accuracy = tf.constant(0., name="dummy_accuracy")
            self.sum_accuracy = tf.constant(0., name="dummy_sum_accuracy")

        # Summaries for loss and accuracy
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.acc_summary = tf.summary.scalar("accuracy", self.accuracy)
        self.aux_loss_summary = tf.summary.scalar("aux_loss", self.aux_loss)
        self.aux_accuracy_summary = tf.summary.scalar("aux_accuracy", self.aux_accuracy)

        vars_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(var_list=vars_to_restore, max_to_keep=None)

        # Train Summaries at Tensorboard
        self.train_summary_op = tf.summary.merge([self.loss_summary, self.acc_summary, self.aux_loss_summary,
                                                  self.aux_accuracy_summary])

        if self.is_primary_model:
            # Define the file writer for tensorboard
            self.train_writer = tf.summary.FileWriter(self.train_log_dir + "train_tensorboard", tf.get_default_graph())

        print('Completed %s object construction.' % self.model_name.name)

    def _train_SGD_batch_step(self, sess, batch_image, batch_label, batch_ids, text_train, train_mode=True,
                              batch_step=1):
        multi_choice_text = text_batch_from_ids(np.reshape(self.aux_text_train_ids[batch_ids], -1),
                                                text_data=text_train)
        S, _, loss_val, acc_val = sess.run(
            [self.train_summary_op, self.train_op, self.aux_loss, self.aux_accuracy],
            feed_dict={
                self.base_image_model.images_placeholder: batch_image,
                self.base_image_model.train_mode: train_mode,
                self.train_mode: train_mode,
                self.aux_labels_placeholder: self.aux_train_labels[batch_ids],
                self.dropout_keep_prob: 0.5,
                self.base_text_model.dropout_keep_prob: 0.5,
                self.base_text_model.texts_placeholder: multi_choice_text,
                self.base_text_model.embedding_dropout: 0.3,
                self.base_text_model.embedding_noise_std: 0.01,
                self.base_text_model.train_mode: train_mode,
                self.base_text_model.phase_train: train_mode,
            })

        return S, loss_val, acc_val

    def _validation_batch_step(self, sess, batch_image, batch_label, batch_ids, text_val, train_mode=False,
                               batch_step=None):
        multi_choice_text = text_batch_from_ids(np.reshape(self.aux_text_val_ids[batch_ids], -1),
                                                text_data=text_val)
        val_aux_acc = sess.run(
            self.aux_sum_accuracy,
            feed_dict={
                self.base_image_model.images_placeholder: batch_image,
                self.base_image_model.train_mode: train_mode,
                self.train_mode: train_mode,
                self.aux_labels_placeholder: self.aux_val_labels[batch_ids],
                self.dropout_keep_prob: 1.,
                self.base_text_model.dropout_keep_prob: 1.,
                self.base_text_model.texts_placeholder: multi_choice_text,
                self.base_text_model.embedding_dropout: 0.,
                self.base_text_model.embedding_noise_std: 0.,
                self.base_text_model.train_mode: train_mode,
                self.base_text_model.phase_train: train_mode,
            })

        return val_aux_acc

    def _test_batch_step(self, sess, c_mat, batch_image, batch_label, batch_ids, text_test):
        # No test set for auxiliary tasks
        pass


class HieCoAtt(ImageTextModel):
    """Hierarchical Question-Image Co-Attention for Visual Question Answering.
    (https://arxiv.org/abs/1606.00061)

    Orignial version by Hsin Wen Liu from Yahoo! Japan internship

    Adapted into this project's structure by Avikalp Srivastava
    """
    def __init__(self, is_primary_model, is_trainable, dim_k=256):
        self.model_name = ModelName.hie_co_att
        self.config = {
            'TRAIN_RESNET_LAST_BLOCK': TRAIN_RESNET_LAST_BLOCK,
            'activation': tf.nn.tanh,
            'base_image_model': ModelName.resnet,
            'base_text_activation': tf.nn.tanh,
            'base_text_filter_sizes': (1, 2, 3),
            'base_text_num_filters': (100, 100, 100),
            'adaptive': False,
            'dim_k': dim_k,
            'use_conv_for_img_reduction': True,
            'use_dropout_on_init_embeddings': True,
        }
        self.activation = self.config['activation']
        self.initializer_type = get_initializer_type(self.activation)
        self.base_text_model_name = self.config['base_text_model']
        self.base_image_model_last_layer = self.base_image_model.representation

        with tf.variable_scope(self.model_name.name):
            super(HieCoAtt, self).__init__(model_name=self.model_name, config=self.config, is_trainable=is_trainable)

            # Construct the text model
            self.base_text_model = HieText(is_primary_model=False,
                                           is_trainable=is_trainable,
                                           filter_sizes=self.config['base_text_filter_sizes'],
                                           num_filters=self.config['base_text_num_filters'],
                                           activation=self.config['base_text_activation'])

            # Deciding what dimensional image embedding we'd get depending on the base image model
            image_dim, image_depth = self._get_derived_image_representation_dimensions()
            self.base_image_model_last_layer = self.base_image_model.representation
            text_embedding_dimension = self.base_text_model.final_embedding_dimension
            dim_d = text_embedding_dimension
            dim_m = image_dim * image_dim
            dim_D = image_depth
            self.dim_k = dim_k
            activation = self.activation
            print("For HieCoAtt: dim_d = %d, dim_m = %d, dim_D = %d" % (dim_d, dim_m, dim_D))

            with tf.variable_scope('dimD_to_dimk'):

                self.v_spI = self._dim_reduction_with_one_one_conv(image_dim, dim_D, dim_k, is_trainable)
                v_spI_unfolded = tf.reshape(self.v_spI, shape=[-1, dim_m, dim_k])
                self.v_I = tf.reduce_mean(v_spI_unfolded, axis=1)

            # Without this layer, performance is worse
            with tf.variable_scope("text_dimk"):
                W_t = weight_variable(is_trainable=is_trainable, shape=[EMBED_SIZES, dim_k],
                                      initializer_type=self.initializer_type, name='W_t')
                b_t = bias_variable(is_trainable=is_trainable, shape=[dim_k], name='b_t')

                word_level = tf.nn.xw_plus_b(tf.reshape(self.base_text_model.word_level, [-1, EMBED_SIZES]), W_t, b_t)
                phrase_level = tf.nn.xw_plus_b(tf.reshape(self.base_text_model.phase_level, [-1, EMBED_SIZES]),
                                               W_t, b_t)
                sentence_level = tf.nn.xw_plus_b(tf.reshape(self.base_text_model.sentence_level,
                                                            [-1, EMBED_SIZES]), W_t, b_t)

                word_level = tf.reshape(tf.nn.dropout(word_level, self.dropout_keep_prob),
                                        [-1, TEXT_LENTH, dim_k])
                phrase_level = tf.reshape(tf.nn.dropout(phrase_level, self.dropout_keep_prob),
                                          [-1, TEXT_LENTH, dim_k])
                sentence_level = tf.reshape(tf.nn.dropout(sentence_level, self.dropout_keep_prob),
                                            [-1, TEXT_LENTH, dim_k])

            # [B, d]
            v_w, q_w, self.att_q_w, self.att_v_w = self._co_attention(word_level,
                                                                      self.v_spI, dim_m, is_trainable, level='word',
                                                                      activation=activation)

            with tf.variable_scope("fc1"):
                W_1 = weight_variable(is_trainable=is_trainable, shape=[dim_k, dim_k],
                                      initializer_type=self.initializer_type, name='W_1')
                b_1 = bias_variable(is_trainable=is_trainable, shape=[dim_k], name='b_1')
                if use_batch_norm:
                    self.h_w = batch_norm_dense_activation(tf.nn.xw_plus_b(x=v_w + q_w, weights=W_1, biases=b_1),
                                                           is_training=self.train_mode, activation=activation,
                                                           is_trainable=is_trainable)
                else:
                    self.h_w = activation(tf.nn.xw_plus_b(x=v_w + q_w, weights=W_1, biases=b_1, name='h_w'))
                self.h_w = tf.nn.dropout(self.h_w, self.dropout_keep_prob)

            # [B, d]
            v_p, q_p, self.att_q_p, self.att_v_p = self._co_attention(phrase_level,
                                                                      self.v_spI, dim_m, is_trainable, level='phrase',
                                                                      activation=activation)
            # [B, 2d]
            with tf.variable_scope("fc2"):
                W_2 = weight_variable(is_trainable=is_trainable, shape=[2 * dim_k, 2 * dim_k],
                                      initializer_type=self.initializer_type, name='W_2')
                b_2 = bias_variable(is_trainable=is_trainable, shape=[2 * dim_k], name='b_2')
                if use_batch_norm:
                    self.h_p = batch_norm_dense_activation(tf.nn.xw_plus_b(x=tf.concat([v_p + q_p, self.h_w], 1),
                                                                           weights=W_2, biases=b_2),
                                                           is_training=self.train_mode,
                                                           activation=activation, is_trainable=is_trainable)
                else:
                    self.h_p = activation(tf.nn.xw_plus_b(x=tf.concat([v_p + q_p, self.h_w], 1),
                                                          weights=W_2, biases=b_2, name='h_p'))
                self.h_p = tf.nn.dropout(self.h_p, self.dropout_keep_prob)

            v_s, q_s, self.att_q_s, self.att_v_s = self._co_attention(sentence_level,
                                                                      self.v_spI, dim_m, is_trainable, level='sentence',
                                                                      activation=activation)
            # [B, 3d]
            with tf.variable_scope("fc3"):
                W_3 = weight_variable(is_trainable=is_trainable, shape=[3 * dim_k, 3 * dim_k],
                                      initializer_type=self.initializer_type, name='W_3')
                b_3 = bias_variable(is_trainable=is_trainable, shape=[3 * dim_k], name='b_3')
                self.h_s = activation(tf.nn.xw_plus_b(x=tf.concat([v_s + q_s, self.h_p], 1),
                                                      weights=W_3, biases=b_3, name='h_s'))
                self.h_s = tf.nn.dropout(self.h_s, self.dropout_keep_prob)

            if is_trainable:
                with tf.name_scope("output_hie"):
                    W_out = weight_variable(is_trainable=is_trainable, shape=[3 * dim_k, NUM_CLASSES],
                                            name='W_out')
                    b_out = bias_variable(is_trainable=is_trainable, shape=[NUM_CLASSES], name='b_out')
                    self.scores = tf.nn.xw_plus_b(self.h_s, W_out, b_out, name="scores_hie")  # unnormalized scores

            if is_primary_model:
                with tf.variable_scope("optimization"):
                    self._set_predictions_optimizer_and_loss()

    def _co_attention(self, texts, images, image_n, is_trainable, level, activation):
        with tf.name_scope("co-att"):
            # V:[B, N, image_dim] Q:[B, T, text_dim]
            # W_b:[B, text_dim, image_dim] # text_dim=image_dim=dim
            # affinity matrix =  relu(Q W V^T) #[B, T, N]
            W_b = weight_variable(is_trainable=is_trainable, shape=[self.dim_k, self.dim_k],
                                  initializer_type=self.initializer_type, name='W_b_%s' % level)

            # [B*T, dim]
            texts_ = tf.reshape(texts, [-1, self.dim_k])
            # [B*N, dim]
            images_ = tf.reshape(images, [-1, self.dim_k])

            # [B*T, dim] --> [B, T, D]
            Q_W_b = tf.reshape(tf.matmul(texts_, W_b), [-1, TEXT_LENTH, self.dim_k])

            # [batch_size,text_length,image_n] # [B, T, D]*[B, D, N]
            c = activation(tf.matmul(Q_W_b, tf.transpose(images, perm=[0, 2, 1])))

            # W_v: [B, k, dim], W_q: [B, k, dim], W_hv: [B, k], W_hq: [B, k]
            # set k = 512
            W_v = weight_variable(is_trainable=is_trainable, shape=[512, self.dim_k],
                                  initializer_type=self.initializer_type, name='W_v_%s' % level)
            W_q = weight_variable(is_trainable=is_trainable, shape=[512, self.dim_k],
                                  initializer_type=self.initializer_type, name='W_q_%s' % level)
            W_hv = weight_variable(is_trainable=is_trainable, shape=[512, 1], initializer_type=self.initializer_type,
                                   name='W_hv_%s' % level)
            W_hq = weight_variable(is_trainable=is_trainable, shape=[512, 1],
                                   initializer_type=self.initializer_type, name='W_hq_%s' % level)

            # [k,D] * [D, B*N] = [k, B*N] --> [B*N,k] --> [B,N,k]
            V_W_v = tf.reshape(tf.transpose(tf.matmul(W_v, tf.transpose(images_))), [-1, image_n, 512])
            # [k,D] * [D, B*T] = [k, B*T] --> [B*T,k] --> [B,T,k]
            Q_W_q = tf.reshape(tf.transpose(tf.matmul(W_q, tf.transpose(texts_))), [-1, TEXT_LENTH, 512])

            # [B,N,k]+ ([B,k,T]*[B,T,N]  = [B, k, N] --> [B,N,k])
            h_v = activation(V_W_v + tf.transpose(tf.matmul(tf.transpose(Q_W_q, perm=[0, 2, 1]), c), perm=[0, 2, 1]))

            # [B*N,k] * [k, 1] = [B*N, 1] = [B,N, 1]
            att_v = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(h_v, [-1, 512]), W_hv), [-1, image_n, 1]))

            # [B, k, T]
            # [B,T,k] +  [B, k, N] *[B, N, T]=[B,k,T] --> [B, T, k]

            h_q = activation(Q_W_q + tf.transpose(tf.matmul(tf.transpose(V_W_v, perm=[0, 2, 1]),
                                                            tf.transpose(c, perm=[0, 2, 1])), perm=[0, 2, 1]))
            # [B*T,k] * [k, 1] = [B*T, 1] = [B,T, 1]
            att_q = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(h_q, [-1, 512]), W_hq), [-1, TEXT_LENTH, 1]))

            # [B, d]?
            v = tf.nn.dropout(tf.reduce_sum(tf.multiply(att_v, images), axis=1), self.dropout_keep_prob)
            q = tf.nn.dropout(tf.reduce_sum(tf.multiply(att_q, texts), axis=1), self.dropout_keep_prob)

        return v, q, att_q, att_v

    def _train_SGD_batch_step(self, sess, batch_image, batch_label, batch_ids, text_train, train_mode=True,
                              batch_step=1):
        S, _, loss_val, acc_val = sess.run(
            [self.train_summary_op, self.train_op, self.loss, self.accuracy],
            feed_dict={
                self.train_mode: train_mode,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 0.5,
                self.base_image_model.images_placeholder: batch_image,
                self.base_image_model.train_mode: train_mode,
                self.base_text_model.texts_placeholder: text_batch_from_ids(batch_ids, text_data=text_train),
                self.base_text_model.embedding_dropout: 0.3,
                self.base_text_model.embedding_noise_std: 0.01,
                self.base_text_model.dropout_keep_prob: 0.5,
                self.base_text_model.train_mode: train_mode,
                self.base_text_model.phase_train: train_mode,
            })

        return S, loss_val, acc_val

    def _validation_batch_step(self, sess, batch_image, batch_label, batch_ids, text_val, train_mode=False,
                               batch_step=1):
        val_acc = sess.run(
            self.sum_accuracy,
            feed_dict={
                self.train_mode: train_mode,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 1.,
                self.base_image_model.images_placeholder: batch_image,
                self.base_image_model.train_mode: True,
                self.base_text_model.texts_placeholder: text_batch_from_ids(batch_ids, text_data=text_val),
                self.base_text_model.embedding_dropout: 0.,
                self.base_text_model.embedding_noise_std: 0.,
                self.base_text_model.dropout_keep_prob: 1.,
                self.base_text_model.train_mode: train_mode,
                self.base_text_model.phase_train: train_mode,
            })

        return val_acc

    def _test_batch_step(self, sess, c_mat, batch_image, batch_label, batch_ids, text_test,
                         train_mode=False):
        pred, mat, test_acc = sess.run(
            [self.probabilities, c_mat, self.sum_accuracy],
            feed_dict={
                self.train_mode: train_mode,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 1.,
                self.base_image_model.images_placeholder: batch_image,
                self.base_image_model.train_mode: True,
                self.base_text_model.texts_placeholder: text_batch_from_ids(batch_ids, text_data=text_test),
                self.base_text_model.embedding_dropout: 0.,
                self.base_text_model.embedding_noise_std: 0.,
                self.base_text_model.dropout_keep_prob: 1.,
                self.base_text_model.train_mode: train_mode,
                self.base_text_model.phase_train: train_mode,
            })

        return pred, mat, test_acc
