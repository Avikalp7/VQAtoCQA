"""
Module for performing answerer retrieval using text-only models
"""


import os

import tensorflow as tf
import numpy as np

from base_model import BaseModel
from text_models import TextCNN
from global_hyperparams import ModelType, ModelName, USE_2014_DATA, ANSWERER_EMBED_DIM, NUM_ANSWERERS, \
    best_epochs, best_date, decay_learning_rate, early_stopping_learning_rate, DECAY_RATE_LEARNING_RATE, \
    DECAY_STEP_LEARNING_RATE, OPTIMIZER, OPTIMIZER_MOMENTUM, OptimizerType, USE_GRADIENT_CLIPPING, \
    batch_size_dict, training_epochs_dict, use_batch_norm
from paths import get_stored_checkpoint_filename
from network import weight_variable, bias_variable
from utils import get_dataset_iterator_from_tfrecords_answerer, update_train_batch, update_val_batch, \
    training_epoch_finish_routine, batch_norm_dense_activation, text_batch_generator


class AnswererModel(BaseModel):

    def restore_base_models(self, sess):
        if self.base_text_model.model_name == ModelName.text_cnn:
            text_model_ckpt_filename = get_stored_checkpoint_filename(
                model_type=self.base_text_model.model_type,
                model_name=self.base_text_model.model_name,
                date=best_date[self.base_text_model.model_name],
                num_epochs=best_epochs[self.base_text_model.model_name]
            )
            vars_to_restore = \
                [v for v in tf.global_variables()
                 if v.name.split('/')[0] == self.base_text_model.model_name.name]
            saver = tf.train.Saver(vars_to_restore)
            # self.global_variable_init = tf.global_variables_initializer()
            self.restore_model_from_filename(sess=sess, model_filename=text_model_ckpt_filename, saver=saver)
        else:
            raise ValueError("Unrecognized base image-text model")

    def __init__(self, base_model_name, is_trainable, is_primary_model=True, config=None):
        assert USE_2014_DATA

        model_name = ModelName.answerer
        activation = tf.nn.tanh
        weight_initializer = 'normal' if activation == tf.nn.relu else 'xavier'

        if base_model_name == ModelName.text_cnn:
            self.base_text_model = TextCNN(filter_sizes=(1, 2, 3),
                                           num_filters=(100, 200, 200),
                                           activation=tf.nn.relu, is_primary_model=False, is_trainable=False)
        else:
            raise ValueError("Not implemented")

        with tf.variable_scope(model_name.name):
            super(AnswererModel, self).__init__(model_type=ModelType.answerer, model_name=model_name,
                                                is_primary_model=is_primary_model, config=config)

            self.labels_placeholder = tf.placeholder(dtype="float", shape=[None, NUM_ANSWERERS],
                                                     name='labels_placeholder')

            self.answerer_embedding_matrix = tf.Variable(
                initial_value=tf.random_uniform([ANSWERER_EMBED_DIM, NUM_ANSWERERS], -0.5, 0.5),
                trainable=is_trainable
            )

            if self.base_text_model.model_name == ModelName.text_cnn:
                self.question_embedding = self.base_text_model.h_pool_flat
            else:
                raise ValueError("Not implmeneted")

            if self.question_embedding.shape[1].value != ANSWERER_EMBED_DIM:
                with tf.variable_scope("ques_dim_to_ans_dim"):
                    W_dim = weight_variable(is_trainable=is_trainable,
                                            shape=[self.question_embedding.shape[1].value, ANSWERER_EMBED_DIM],
                                            initializer_type=weight_initializer,
                                            name='W_dim')
                    b_dim = bias_variable(is_trainable=is_trainable,
                                          shape=[ANSWERER_EMBED_DIM],
                                          name='b_dim')
                    self.question_embedding = tf.nn.xw_plus_b(x=self.question_embedding, weights=W_dim,
                                                              biases=b_dim, name='reduced_question_embedding')

                    if use_batch_norm:
                        self.question_embedding = batch_norm_dense_activation(inputs=self.question_embedding,
                                                                              is_training=self.train_mode,
                                                                              activation=activation,
                                                                              is_trainable=is_trainable)
                    else:
                        self.question_embedding = activation(self.question_embedding)

            self.similarity_matrix = weight_variable(
                is_trainable=is_trainable,
                shape=[self.question_embedding.shape[1].value, self.answerer_embedding_matrix.shape[0].value],
                initializer_type=weight_initializer, name='similarity_matrix'
            )
            # self.similarity_biases = bias_variable(is_trainable=is_trainable, shape=[NUM_ANSWERERS])

            intermed_matrix = tf.matmul(self.question_embedding, self.similarity_matrix)
            self.scores = tf.matmul(intermed_matrix, self.answerer_embedding_matrix)

            with tf.variable_scope("optimization"):
                self._set_predictions_optimizer_and_loss()

    def _set_predictions_optimizer_and_loss(self):
        self.probabilities = tf.nn.sigmoid(self.scores, name="probabilities")  # normalized scores

        # if USE_MULTILABEL:
        #     self.predictions = tf.round(self.probabilities, name="predictions")
        # else:
        #     self.predictions = tf.argmax(self.probabilities, 1, name="predictions")

        if self.trainable:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels_placeholder,
                                                                               logits=self.scores))

        if decay_learning_rate:
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                            DECAY_STEP_LEARNING_RATE, DECAY_RATE_LEARNING_RATE,
                                                            staircase=True)
        elif early_stopping_learning_rate:
            self.learning_rate = tf.Variable(self.learning_rate, trainable=False, name='learning_rate_var')

        if self.model_type != ModelType.image_only:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_name.name)
        else:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # if self.model_type == ModelType.image_text:
        #     update_ops = tf.get_collection(UPDATE_OPS_COLLECTION) + update_ops

        with tf.control_dependencies(update_ops):
            # Define Training procedure
            if OPTIMIZER == OptimizerType.adam:
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif OPTIMIZER == OptimizerType.rms_optimizer:
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                           momentum=OPTIMIZER_MOMENTUM)
            else:
                raise ValueError("Invalid value for OPTIMIZER var")

            if USE_GRADIENT_CLIPPING:
                # gradient clipping
                gvs = self.optimizer.compute_gradients(self.loss, tf.trainable_variables())
                with tf.device('/cpu:0'):
                    clipped_gvs = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs if grad is not None]
                self.train_op = self.optimizer.apply_gradients(clipped_gvs, global_step=self.global_step)
            else:
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # Calculate MRR
        with tf.variable_scope("MRR"):
            argsort = tf.cast(tf.nn.top_k(self.probabilities, k=NUM_ANSWERERS, sorted=True).indices, dtype="float")
            argsort = tf.cast(tf.nn.top_k(argsort, k=NUM_ANSWERERS, sorted=True).indices, dtype="float")
            rev_argsort = tf.reverse(argsort, axis=[-1])
            temp_mul = (float(NUM_ANSWERERS) - rev_argsort) * self.labels_placeholder
            self.best_ranks = 1. + (float(NUM_ANSWERERS) - tf.reduce_max(temp_mul, axis=-1))
            # temp_mul = (1. + argsort) * (1. - self.labels_placeholder)
            # self.best_ranks = tf.cast(tf.argmin(temp_mul, axis=1, name='best_ranks'), dtype="float")
            self.inverse_ranks = 1. / (1. + self.best_ranks)
            self.sum_rr = tf.reduce_sum(self.inverse_ranks, name="sum_rr")
            self.mrr = tf.reduce_mean(self.inverse_ranks, name="mrr")

        # Summaries for loss and accuracy
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.mrr_summary = tf.summary.scalar("mrr", self.mrr)

        vars_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(var_list=vars_to_restore, max_to_keep=None)

        # Train Summaries at Tensorboard
        self.train_summary_op = tf.summary.merge([self.loss_summary, self.mrr_summary])

        # Define the file writer for tensorboard
        if self.is_primary_model:
            self.train_writer = tf.summary.FileWriter(self.train_log_dir + "train_tensorboard", tf.get_default_graph())

        print('Completed %s object construction.' % self.model_name.name)

    def train(self, train_ids, train_labels, train_texts, val_ids, val_labels, val_texts, retrain_model_filename=None):
        """ Train the model on given data, saving model checkpoints and log files.
        :param train_ids: np.array, self-explanatory
        :param train_labels: np.array
        :param train_texts: np.array of np.arrays
        :param val_ids: np.array
        :param val_labels: np.array
        :param val_texts: np.array of np.arrays
        :param retrain_model_filename: str, if None, model is trained from scratch, else this param states where the
            ckpt file resides, and model is further trained from that ckpt
        :return: None
        """
        # iterator_train = get_dataset_iterator_from_tfrecords_answerer(data_type='train',
        #                                                               batch_size=self.batch_size)
        # iterator_val = get_dataset_iterator_from_tfrecords_answerer(data_type='validation',
        #                                                             batch_size=self.batch_size)
        #
        # iterator_train_init = iterator_train.initializer
        # iterator_val_init = iterator_val.initializer
        # next_element_train = iterator_train.get_next()
        # next_element_val = iterator_val.get_next()

        iterator_train_init, next_element_train, iterator_val_init, next_element_val = None, None, None, None

        vars_to_restore = tf.global_variables()
        saver = tf.train.Saver(vars_to_restore)
        self.global_variable_init = tf.global_variables_initializer()

        sess = self._train_setup(train_ids, val_ids, batch_size=batch_size_dict[self.model_type])

        if retrain_model_filename is not None:
            self.restore_model_from_filename(sess, model_filename=retrain_model_filename, saver=saver)
        else:
            self.restore_base_models(sess)

        self._train_helper_SGD(sess, self.train_writer, train_ids, train_labels,
                               train_texts, val_ids, val_labels, val_texts, iterator_train_init, next_element_train,
                               iterator_val_init, next_element_val)

        # Final global save
        print("\nSaving final model .ckpt file ...")
        _ = self.saver.save(sess, self.checkpoint_dir + "model.ckpt")
        print("Done saving final model .ckpt file.")

        # Close tensorflow session
        sess.close()

    def test(self, date, num_epochs, test_ids, test_labels, test_texts):
        print("\nStarting to test %s model" % self.model_name.name)
        print("Test Data size: %d" % len(test_ids))

        if self.model_type != ModelType.text_only:
            iterator_test = get_dataset_iterator_from_tfrecords_answerer(data_type='test',
                                                                         batch_size=self.batch_size)
            iterator_test_init = iterator_test.initializer
            next_element_test = iterator_test.get_next()
        else:
            iterator_test, iterator_test_init, next_element_test = None, None, None

        vars_to_restore = tf.global_variables()
        saver = tf.train.Saver(vars_to_restore)
        # self.global_variable_init = tf.global_variables_initializer()
        # tf.get_default_graph().finalize()

        # Make directory to write results to
        if not os.path.exists(self.test_log_dir):
            os.makedirs(self.test_log_dir)

        with tf.Session() as sess:
            # TODO: RM
            self.restore_model(sess, date, num_epochs, saver)
            # Run the tests and get accuracy, predicted labels, confusion matrix etc.
            test_accuracy, labels, results, conf_matrix = self._run_tests(sess, test_ids, test_labels, test_texts,
                                                                          iterator_test_init, next_element_test)
            self._write_results_to_file(conf_matrix, date, labels, results, test_ids, test_accuracy)

    # noinspection PyTypeChecker
    def _train_helper_SGD(self, sess, train_writer, train_ids, train_labels, train_texts,
                          val_ids, val_labels, val_texts, iterator_train_init, next_element_train, iterator_val_init,
                          next_element_val):
        """ Helper function for train(), implementing the SGD steps over train & val data in arguments. """
        global_step = sess.run(self.global_step)

        # EARLY STOPPING VARS
        val_acc_history = []
        epochs_with_current_lrate = 0
        min_increment = 0.0002
        tolerable_decrease = 0.0004

        batch_size = batch_size_dict[self.model_type]
        training_epochs = training_epochs_dict[self.model_type]

        for epoch in range(training_epochs):
            # Decay learning rate if required
            if early_stopping_learning_rate:
                if epochs_with_current_lrate >= 3 and val_acc_history[-1] < (val_acc_history[-3] + min_increment):
                    best_epoch = np.argmax(val_acc_history)
                    self.restore_mid_training(sess=sess, num_epochs=best_epoch)
                    if len(val_acc_history) >= 8 and val_acc_history[-1] < (val_acc_history[-8] + min_increment):
                        self.learning_rate.assign(self.learning_rate / 10.).eval(session=sess)
                    else:
                        self.learning_rate.assign(self.learning_rate / 2.).eval(session=sess)
                    epochs_with_current_lrate = 0
                elif epochs_with_current_lrate >= 1 and \
                        (max(val_acc_history) - val_acc_history[-1]) > tolerable_decrease:
                    best_epoch = np.argmax(val_acc_history)
                    self.restore_mid_training(sess=sess, num_epochs=best_epoch)
                    self.learning_rate.assign(self.learning_rate / 2.).eval(session=sess)

            if len(val_acc_history) > 15 and val_acc_history[-1] < (val_acc_history[-15] + min_increment):
                print('Early breaking')
                break

            total_epoch_acc_val = 0.
            gen = text_batch_generator(batch_size, train_texts, train_labels)

            total_batches = int(len(train_texts) / batch_size) + 1
            print("*** TOTAL BATCHES: %d ***" % total_batches)
            # Completed training for this epoch
            print("\nCurrent learning rate: {}".format(sess.run(self.learning_rate)))

            for i in range(total_batches):
                batch_text, batch_label = gen.next()

                S, loss_val, acc_val = self._train_SGD_batch_step(sess, None, batch_label,
                                                                  None, batch_text, batch_step=global_step)
                global_step, total_epoch_acc_val = update_train_batch(global_step, S, loss_val, acc_val, train_writer,
                                                                      total_epoch_acc_val,
                                                                      batches_completed_this_epoch=i + 1)

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
                val_acc_sum, num_steps = update_val_batch(num_steps, val_acc, val_acc_sum, total_batches)

            # num_samples = len(val_ids)
            training_epoch_finish_routine(sess, val_acc_sum, num_samples, self.train_logfile_name,
                                          self.checkpoint_dir, epoch, self.saver)
            val_acc_history.append(val_acc_sum / float(num_samples))
            print ('Previous Val Accs:', val_acc_history[-10:])
            epochs_with_current_lrate += 1

        # Completed all epochs
        print('BEST VAL: ', max(val_acc_history))
        print("Completed training %s model, with %d epochs, each with %d minibatches each of size %d"
              % (self.model_name.name, training_epochs + 1, global_step + 1, batch_size))

    def _train_SGD_batch_step(self, sess, batch_image, batch_label, batch_ids, text_train, train_mode=True,
                              batch_step=1):
        _ = batch_step
        S, _, loss_val, acc_val = sess.run(
            [self.train_summary_op, self.train_op, self.loss, self.mrr],
            feed_dict={
                self.train_mode: train_mode,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 0.5,

                self.base_text_model.texts_placeholder: text_train,
                self.base_text_model.embedding_dropout: 0.,
                self.base_text_model.embedding_noise_std: 0.,
                self.base_text_model.train_mode: False,
                self.base_text_model.phase_train: False,
            })

        return S, loss_val, acc_val

    def _validation_batch_step(self, sess, batch_image, batch_label, batch_ids, text_val, train_mode=False,
                               batch_step=1):
        _ = batch_step
        val_acc = sess.run(
            self.sum_rr,
            feed_dict={
                self.train_mode: train_mode,
                self.labels_placeholder: batch_label,
                self.dropout_keep_prob: 1.,

                self.base_text_model.texts_placeholder: text_val,
                self.base_text_model.embedding_dropout: 0.,
                self.base_text_model.embedding_noise_std: 0.,
                self.base_text_model.train_mode: False,
                self.base_text_model.phase_train: False,
            })

        return val_acc
