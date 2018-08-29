"""Module implementing text based models for CQA tasks

This module is responsible for:

    * Implementing the base class TextModel from which all other text models will be derived

    * Implementing the text models: TextCNN (based on (Kim, 2014)), TextRNN, and HieText (based on (Lu et al. 2016)

    * Implementing the training, validation and test procedures for each model
"""
# Copyright (C) 2018 Yahoo Japan Corporation (Licensed under CC BY-NC-SA 4.0)
# https://creativecommons.org/licenses/by-nc-sa/4.0/


import tensorflow as tf
from tensorflow.contrib import rnn

from base_model import BaseModel
from utils import batch_norm, update_train_batch, update_val_batch, update_test_batch, \
    test_finish_routine, training_epoch_finish_routine, right_align, text_batch_generator
from network import weight_variable, bias_variable
from global_hyperparams import NUM_CLASSES, ModelType, ModelName, batch_size_dict, \
    training_epochs_dict, TEXT_LENTH, WORD_SIZES, EMBED_SIZES, USE_MULTILABEL


class TextModel(BaseModel):
    """Base text model

        * Contains definitions for text placeholder and embedding matrix and lookup procedure.

        * Implements a common train and test routine for all.
    """

    def __init__(self, model_name, is_trainable,
                 form_embedding_matrix=True, is_primary_model=True, config=None, embed_size_multiplier=1.):
        """Calls BaseModel constructor, declares placeholders, embedding matrix lookup procedure.

        Parameters
        ----------
        :param model_name: ModelName enum

        :param is_trainable: bool, whether the model's weight params can be trained/modified

        :param form_embedding_matrix: bool, whether to setup an embedding matrix and lookup architecture

        :param is_primary_model: bool, whether to setup optimization variables for this model.
            If it's embedding output is simply fed to some other image_text model, then it is False.

        :param config: dict, containing configuration parameters

        :param embed_size_multiplier: float, used to explicitly scale embedding_size
        """
        # Call the BaseModel constructor
        super(TextModel, self).__init__(model_type=ModelType.text_only, model_name=model_name,
                                        is_primary_model=is_primary_model, config=config)

        # Placeholder for text data, shape: [batch_size, TEXT_LENGTH]
        self.texts_placeholder = tf.placeholder("int32", shape=[None, TEXT_LENTH], name='text_placeholder')
        # Placeholder indicating whether model is training or validating/testing
        self.phase_train = tf.placeholder(tf.bool, name='phase_train_placeholder')
        # Placeholders for purpose of regularization
        self.embedding_dropout = tf.placeholder_with_default(0., shape=[], name='embedding_dropout_placeholder')
        self.embedding_noise_std = tf.placeholder_with_default(0., shape=[], name='embedding_noise_placeholder')

        # The embedding size is mentioned in global_hyperparams. User can explicitly scale it via embed_size_multiplier
        self.embedding_size = int(EMBED_SIZES * embed_size_multiplier)

        if form_embedding_matrix:
            # Embedding layer
            with tf.variable_scope("embedding"):
                # Matrix with embedding for each word in vocab
                self.embedding_matrix = \
                    tf.Variable(
                        trainable=is_trainable,
                        initial_value=tf.random_uniform([WORD_SIZES, self.embedding_size], -0.5, 0.5))
                # Lookup to form embedding sequence for current text input
                self.embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.texts_placeholder)
                # Add noise and dropout
                self.embedded_noise = self.embedded + tf.random_normal(shape=tf.shape(self.embedded),
                                                                       stddev=self.embedding_noise_std)
                self.seq_embedded = tf.layers.dropout(self.embedded_noise, rate=0.,
                                                      noise_shape=tf.shape(self.embedded[:-1] + [1]))

            # input for convolution: 4D tensor: [batch_size, text_size, embedding_size, channel_size]
            self.x_text = tf.reshape(self.seq_embedded, [-1, TEXT_LENTH, self.embedding_size, 1])
        else:
            self.embedding_matrix, self.embedded, self.seq_embedded, self.x_text = None, None, None, None

    def _train_helper_SGD(self, sess, train_writer, train_ids, train_labels, train_texts,
                          val_ids, val_labels, val_texts, iterator_train_init, next_element_train, iterator_val_init,
                          next_element_val):
        """Helper function for train(), implementing the SGD steps over train & val data in arguments.

        Parameters
        ----------
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
        global_step = sess.run(self.global_step)

        # For RNN, all texts are right aligned
        if self.model_name == ModelName.text_rnn:
            train_texts = right_align(train_texts)
            val_texts = right_align(val_texts)

        # Variable for implementing early stopping and validation acc based learning rate decay
        val_acc_history = []
        epochs_with_current_lrate = 0
        min_increment = 0.0002
        tolerable_decrease = 0.0004

        training_epochs = training_epochs_dict[self.model_type]

        for epoch in range(training_epochs):
            # Decay learning rate (if early_stopping_learning_rate=True) on basis of val acc history
            # And determine whether to early stop.
            early_stop, epochs_with_current_lrate = self.early_stopping_procedure(sess,
                                                                                  epochs_with_current_lrate,
                                                                                  val_acc_history,
                                                                                  min_increment, tolerable_decrease)
            if early_stop: break

            # Running training GD for this epoch
            global_step = self._train_SGD_loop(sess, train_texts, train_labels, global_step, train_writer)
            # Completed training for this epoch

            # Now performing validation
            self._val_prediction_loop(sess, epoch, val_texts, val_labels, val_acc_history)

            print ('Previous Val Accs:', val_acc_history[-10:])
            epochs_with_current_lrate += 1

        # Completed all epochs
        print('BEST VAL: ', max(val_acc_history))
        print("Completed training %s model, with %d epochs, each with %d minibatches each of size %d"
              % (self.model_name.name, training_epochs + 1, global_step + 1, self.batch_size))

    def _train_SGD_loop(self, sess, train_texts, train_labels, global_step, train_writer):
        """Mini-batch gradient descent over training data"""
        # Book-keeping vars
        total_batches = int(len(train_texts) / self.batch_size) + 1
        total_epoch_acc_val = 0.
        # gen yields the next text batch at each call of .next()
        gen = text_batch_generator(self.batch_size, train_texts, train_labels)

        print("*** TOTAL BATCHES: %d ***" % total_batches)
        print("\nCurrent learning rate: {}".format(sess.run(self.learning_rate)))

        for i in range(total_batches):
            batch_text, batch_label = gen.next()
            # Run the SGD for current batch, and get the summary values, loss value, and accuracy value
            summary, loss_val, acc_val = self._train_SGD_batch_step(sess=sess, batch_image=None,
                                                                    batch_label=batch_label, batch_ids=None,
                                                                    text_train=batch_text, batch_step=global_step)
            # Update book-keeping vars
            global_step, total_epoch_acc_val = update_train_batch(global_step, summary, loss_val, acc_val,
                                                                  train_writer, total_epoch_acc_val,
                                                                  batches_completed_this_epoch=(i + 1))
        return global_step

    def _val_prediction_loop(self, sess, epoch, val_texts, val_labels, val_acc_history):
        """Run prediction loop over batches from validation

        Parameters
        ----------
        :param sess: tf.Session() object

        :param epoch: int, the current epoch number

        :param val_texts: np.array of shape [NUM_VAL_SAMPLES, TEXT_LENGTH] and type int

        :param val_labels: np.array of shape [NUM_VAL_SAMPLES, NUM_CLASSES] and type bool

        :param val_acc_history: list, containing floats indicating past validation accuracy values
        """
        # Book-keeping vars for validation
        num_val_batches = (len(val_texts) / batch_size_dict[self.model_type]) + 1
        val_acc_sum = 0.
        num_steps = 0
        num_samples = 0
        for i in range(num_val_batches):
            batch = batch_size_dict[self.model_type] * i
            end = batch + batch_size_dict[self.model_type]
            if end > len(val_texts):
                end = len(val_texts)
            num_samples += (end - batch)
            val_acc = self._validation_batch_step(sess, None, val_labels[batch:end], None, val_texts[batch:end],
                                                  batch_step=num_steps, train_mode=False)
            val_acc_sum, num_steps = update_val_batch(num_steps, val_acc, val_acc_sum, num_val_batches)

        # num_samples = len(val_ids)
        training_epoch_finish_routine(sess, val_acc_sum, num_samples, self.train_logfile_name,
                                      self.checkpoint_dir, epoch, self.saver)
        val_acc_history.append(val_acc_sum / float(num_samples))

    def _run_tests(self, sess, test_ids, test_labels, test_texts, iterator_test_init, next_element_test):
        """Run prediction loop over batches from test set"""
        print("\nRunning test data through model ...")

        # Right align text data in case RNN model is used
        if self.model_name == ModelName.text_rnn:
            test_texts = right_align(test_texts)

        # Some required variables
        conf_matrix = None
        prediction = None
        test_accuracy_sum = 0.
        c_mat = tf.contrib.metrics.confusion_matrix(self.predictions, tf.argmax(self.labels_placeholder, 1),
                                                    num_classes=NUM_CLASSES)
        total_batches = int(len(test_labels) / batch_size_dict[self.model_type]) + 1
        batch_num = 0
        num_samples = 0
        for i in range(total_batches):
            batch = batch_size_dict[self.model_type] * i
            end = batch + batch_size_dict[self.model_type]
            # For the last batch
            if end > len(test_texts):
                end = len(test_texts)

            num_samples += (end - batch)
            pred, mat, test_acc = self._test_batch_step(sess, c_mat,
                                                        None, test_labels[batch:end], None, test_texts[batch:end])
            # Update all book-keeping variables
            conf_matrix, prediction, test_accuracy_sum = \
                update_test_batch(batch_num, pred, mat, test_acc, conf_matrix,
                                  prediction, test_accuracy_sum, total_batches)
            batch_num += 1
        # Gather the original labels and predictions
        labels, predictions = test_finish_routine(test_labels, prediction)
        # Average over the summed accuracy
        test_accuracy = test_accuracy_sum / float(len(test_labels))

        return test_accuracy, labels, predictions, conf_matrix

    def _run_validation_test(self, sess, val_ids, val_labels, val_texts, iterator_validation, next_element_val):
        """Run prediction over the validation set, using a pre-trained model"""
        val_acc_sum = 0
        num_val_batches = int(len(val_texts) / batch_size_dict[self.model_type]) + 1
        for i in range(num_val_batches):
            batch = batch_size_dict[self.model_type] * i
            end = batch + batch_size_dict[self.model_type]
            if end > len(val_texts):
                end = len(val_texts)
            val_acc = sess.run(self.sum_accuracy, feed_dict={
                self.texts_placeholder: val_texts[batch:end],
                self.labels_placeholder: val_labels[batch:end],
                self.dropout_keep_prob: 1.0,
                self.phase_train: False
            })
            val_acc_sum += val_acc

        val_acc = val_acc_sum / float(len(val_labels))
        print("*** Validation Accuracy: %0.4f ***" % val_acc)

    def _train_SGD_batch_step(self, sess, batch_image, batch_label, batch_ids, text_train, train_mode=True,
                              batch_step=1):
        """Run the tensorflow's computation graph for the current batch.

        The train_op is computated, which is the optimization method over the weights.

        Returns
        -------
        S: summary for tensorboard
        loss_val: float, calculated loss value for this batch
        acc_val: float, calculated accuracy value for this batch
        """
        _ = batch_step, batch_image, batch_ids

        S, _, loss_val, acc_val = sess.run(
            [self.train_summary_op, self.train_op, self.loss, self.accuracy],
            feed_dict={
                self.texts_placeholder: text_train,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 0.5,
                self.embedding_dropout: 0.3,
                self.embedding_noise_std: 0.01,
                self.phase_train: train_mode
            })

        return S, loss_val, acc_val

    def _validation_batch_step(self, sess, batch_image, batch_label, batch_ids, text_val, train_mode=False,
                               batch_step=1):
        """Run the tensorflow's computation graph for prediction on current batch

        Returns
        -------
        val_acc: float, computed valdation accuracy over the current batch
        """
        _ = batch_image, batch_ids, batch_step

        val_acc = sess.run(self.sum_accuracy, feed_dict={
            self.texts_placeholder: text_val,
            self.labels_placeholder: batch_label,
            self.dropout_keep_prob: 1.0,
            self.phase_train: train_mode
        })

        return val_acc

    def _test_batch_step(self, sess, c_mat, batch_image, batch_label, batch_ids, text_test,
                         train_mode=False):
        _ = batch_image, batch_ids
        pred, mat, test_acc = sess.run([self.probabilities, c_mat, self.sum_accuracy], feed_dict={
            self.texts_placeholder: text_test,
            self.labels_placeholder: batch_label,
            self.dropout_keep_prob: 1.0,
            self.phase_train: train_mode
        })

        return pred, mat, test_acc


class TextCNN(TextModel):
    """A baseline CNN for text only classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    (Kim, 2014)
    """

    def __init__(self, filter_sizes=(1, 2, 3), num_filters=(128, 256, 256), activation=tf.nn.relu,
                 is_primary_model=True, is_trainable=True, embed_size_multiplier=1.):
        """Create an embedding matrix, a lookup procedure on it for text sequence input,
            convoultion using multiple filter sizes, and final optimizations if is_primary_model=True.

        Parameters
        ----------
        :param filter_sizes: tuple, containing the different filter sizes for convolution

        :param num_filters: int or tuple, should be of same length as filter_sizes,
            denoting the number of filter of each size. If int, it implies same number of filter for all sizes.

        :param activation: activation function from tf.nn (only tf.nn.relu, tf.nn.tanh supported)

        :param is_primary_model: bool, whether this model's output is used to perform the task

        :param is_trainable: bool, whether parameters can be updated during training

        :param embed_size_multiplier: float, factor with which to multiply default embedding size
        """
        with tf.variable_scope(ModelName.text_cnn.name):
            super(TextCNN, self).__init__(model_name=ModelName.text_cnn, is_primary_model=is_primary_model,
                                          is_trainable=is_trainable, embed_size_multiplier=embed_size_multiplier)

            # Convert num_filters to list corresponding to number of filters for each filter size
            if isinstance(num_filters, int):
                num_filters = [num_filters] * len(filter_sizes)

            assert len(num_filters) == len(filter_sizes)
            assert activation == tf.nn.relu or activation == tf.nn.tanh

            self.config = {
                'filter_sizes': filter_sizes,
                'num_filters': num_filters,
                'activation': 'relu' if activation == tf.nn.relu else 'tanh'
            }
            # convolution + maxpool
            self.pooled_outputs = []

            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.embedding_size, 1, num_filters[i]]
                    initializer_type = "normal" if activation == tf.nn.relu else "xavier"
                    W_o = weight_variable(is_trainable=is_trainable, shape=filter_shape,
                                          initializer_type=initializer_type,
                                          name='W%d' % i)
                    # pad to prevent dimension reduction
                    twopadding = filter_size - 1  # (h+2p-f)/s + 1 = h #s=1 #same height padding
                    top_padding = twopadding // 2
                    bottom_padding = twopadding - top_padding
                    self.x_padded = tf.pad(self.x_text, [[0, 0], [top_padding, bottom_padding], [0, 0], [0, 0]])
                    # Do convolution + batch_norm + activation
                    conv = tf.nn.conv2d(self.x_padded, W_o, strides=[1, 1, 1, 1], padding='VALID', name="conv")
                    bn_conv = batch_norm(conv, num_filters[i], self.phase_train)
                    h = activation(bn_conv, "activation")

                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, TEXT_LENTH, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")

                    self.pooled_outputs.append(pooled)

            # Combine all the pooled features
            self.final_embedding_dimension = sum(num_filters)
            self.h_pool = tf.concat(self.pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.final_embedding_dimension])

            # Add dropout
            with tf.variable_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            if is_trainable:
                # Final (unnormalized) scores and predictions
                with tf.variable_scope("output"):
                    W_o = weight_variable(is_trainable=is_trainable,
                                          shape=[self.final_embedding_dimension, NUM_CLASSES],
                                          name='W_o', initializer_type='xavier' if USE_MULTILABEL else 'normal')
                    b_o = bias_variable(is_trainable=is_trainable, shape=[NUM_CLASSES], name='b_o')
                    self.scores = tf.nn.xw_plus_b(self.h_drop, W_o, b_o, name="scores")  # unnormalized scores

                if self.is_primary_model:
                    self._set_predictions_optimizer_and_loss()


class HieText(TextModel):
    """
    Hierarchical Question-Image Co-Attention for Visual Question Answering.
    (https://arxiv.org/abs/1606.00061)
    """

    def __init__(self, is_primary_model, is_trainable, filter_sizes=(1, 2, 3), num_filters=128,
                 activation=tf.nn.tanh):
        """Define the TF elements for HieCoAtt's text representation.

        Parameters
        ----------
        :param filter_sizes: tuple, containing the different filter sizes for convolution

        :param num_filters: int, number of filters of each size

        :param activation: activation function from tf.nn (only tf.nn.relu, tf.nn.tanh supported)

        :param is_primary_model: bool, whether this model's output is used to perform the task

        :param is_trainable: bool, whether parameters can be updated during training
        """
        assert activation == tf.nn.relu or activation == tf.nn.tanh
        self.activation = activation
        self.initializer_type = 'xavier' if self.activation == tf.nn.tanh else 'normal'

        with tf.variable_scope(ModelName.hie_text.name):
            super(HieText, self).__init__(model_name=ModelName.hie_text, is_primary_model=is_primary_model,
                                          is_trainable=is_trainable)

            # Convert num_filters to list corresponding to number of filters for each filter size
            if isinstance(num_filters, int):
                num_filters = [num_filters] * len(filter_sizes)
            # Dimension of phrase level features should match dimension of word-level features
            # So, we need that num_filters matches EMBED_SIZES
            assert num_filters == EMBED_SIZES

            self.config = {
                'filter_sizes': filter_sizes,
                'num_filters': num_filters,
                'activation': 'relu' if self.activation == tf.nn.relu else 'tanh'
            }

            # Shape fo self.word_level: [BATCH_SIZE, TEXT_LENGTH, EMBED_SIZES]
            self.word_level = self.seq_embedded

            # Convolution for phrase level
            self.conv_output = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, EMBED_SIZES, 1, num_filters[i]]
                    W = weight_variable(is_trainable=is_trainable, shape=filter_shape,
                                        initializer_type=self.initializer_type,
                                        name='W%d' % i)
                    twopadding = filter_size - 1  # (h+2p-f)/s + 1 = h #s=1 #same height padding
                    top_padding = twopadding // 2
                    bottom_padding = twopadding - top_padding
                    self.x_padded = tf.pad(self.x_text, [[0, 0], [top_padding, bottom_padding], [0, 0], [0, 0]])
                    conv = tf.nn.conv2d(self.x_padded, W, strides=[1, 1, 1, 1], padding='VALID', name="conv")
                    bn_conv = batch_norm(conv, num_filters[i], self.phase_train)
                    h = self.activation(bn_conv, name="activation")
                    self.conv_output.append(h)

            # Shape of full_conv_output: [BATCH_SIZE, TEXT_LENGTH, len(filter_sizes), num_filters]
            full_conv_output = tf.concat(self.conv_output, 2)

            # Phrase level output shape: [BATCH_SIZE, TEXT_LENGTH, num_filters]
            self.phase_level = tf.reduce_max(full_conv_output, 2)

            # Sentence Level
            lstm_cell = rnn.BasicLSTMCell(EMBED_SIZES)
            if self.dropout_keep_prob is not None:
                lstm_cell = rnn.DropoutWrapper(lstm_cell,
                                               output_keep_prob=self.dropout_keep_prob)
            self.lstm_outputs, states = tf.nn.dynamic_rnn(lstm_cell,
                                                          self.phase_level,
                                                          dtype=tf.float32)
            # Sentence_level text output
            # [BATCH_SIZE, TEXT_LENGTH, num_filters]
            self.sentence_level = tf.concat(self.lstm_outputs, 1)

            # Concatenate the different levels.
            # We tried the hierarchical approach in Lu et al., but it gave inferior results.
            self.final_text_embedding_spatial = tf.concat(values=(self.word_level,
                                                                  self.phase_level,
                                                                  self.sentence_level),
                                                          axis=-1)
            self.final_text_embedding = tf.reduce_mean(self.final_text_embedding_spatial, axis=1)
            # Add dropout
            with tf.variable_scope("dropout"):
                self.final_text_embedding = tf.nn.dropout(self.final_text_embedding, self.dropout_keep_prob)
            self.final_embedding_dimension = self.final_text_embedding.shape[1].value

            if is_trainable:
                # Final (unnormalized) scores and predictions
                with tf.variable_scope("output"):
                    W_o = weight_variable(is_trainable=is_trainable,
                                          shape=[self.final_embedding_dimension, NUM_CLASSES],
                                          name='W_o', initializer_type='xavier' if USE_MULTILABEL else 'normal')
                    b_o = bias_variable(is_trainable=is_trainable, shape=[NUM_CLASSES], name='b_o')
                    self.scores = tf.nn.xw_plus_b(self.final_text_embedding, W_o, b_o, name="scores")

                if self.is_primary_model:
                    self._set_predictions_optimizer_and_loss()
