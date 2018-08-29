""" Module implementing the base class model for all our future image-only, text-only, and image-text models.
"""

import os

import numpy as np
import tensorflow as tf

from utils import get_dataset_iterator_from_tfrecords
from paths import image_results_subdirectory, train_log_directory_suffix, answerer_results_subdirectory, \
    checkpoint_directory_suffix, test_log_directory_suffix, image_text_results_subdirectory, \
    text_results_subdirectory, get_stored_checkpoint_filename, get_bottleneck_data_subdirectory, TIME_STAMP
from global_hyperparams import NUM_CLASSES, ModelType, batch_size_dict, decay_learning_rate, \
    learning_rate_dict, NUM_IMAGES, scale_learning_rate, scale_epochs_dict, scale_factors_dict, global_config, \
    DECAY_RATE_LEARNING_RATE, DECAY_STEP_LEARNING_RATE, best_date, best_epochs, IMAGE_SIZE, TEXT_LENTH, EMBED_SIZES, \
    OPTIMIZER, OptimizerType, OPTIMIZER_MOMENTUM, USE_GRADIENT_CLIPPING, early_stopping_learning_rate, USE_MULTILABEL


class BaseModel(object):
    """ Base model class for all text-only, image-only, and image-text based models

        Has abstract method _train_helper_SGD that should implement the SGD step for each derived model.
    """

    def __init__(self, model_type, model_name, is_trainable=True, is_primary_model=True, config=None):
        """Set and define attributes across all models
        
        Parameters
        ----------
        :param model_type: ModelType Enum, 
            ModelType.text_only | ModelType.image_only | ModelType.image_text
        
        :param model_name: ModelName Enum
            Possible values can be seen in the global_hyperparams.py file
        
        :param is_trainable: bool, whether weights can be updated
        
        :param is_primary_model: bool, whether this model's output will be used for the final task
        
        :param config: dict, containing various parameter name-value pairs. Used for writing config file.
        """
        super(BaseModel, self).__init__()
        self.model_type = model_type
        self.model_name = model_name
        self.batch_size = batch_size_dict[model_type]

        # Directories for book-keeping
        if model_type == ModelType.image_only:
            self.results_subdirectory = image_results_subdirectory
        elif model_type == ModelType.text_only:
            self.results_subdirectory = text_results_subdirectory
        elif model_type == ModelType.image_text:
            self.results_subdirectory = image_text_results_subdirectory
        elif model_type == ModelType.answerer:
            self.results_subdirectory = answerer_results_subdirectory
        else:
            raise ValueError('Error: Ill-defined model type %s' % model_type)

        self.config = config
        self.model_type_config = {
            'batch_size': batch_size_dict[self.model_type],
            'learning_rate': learning_rate_dict[self.model_type],
        }
        if scale_learning_rate:
            self.model_type_config['scale_epochs_dict'] = scale_epochs_dict[self.model_type]
            self.model_type_config['scale_factors_dict'] = scale_factors_dict[self.model_type]
        if decay_learning_rate:
            self.model_type_config['DECAY_STEP_LEARNING_RATE'] = DECAY_STEP_LEARNING_RATE
            self.model_type_config['DECAY_RATE_LEARNING_RATE'] = DECAY_RATE_LEARNING_RATE
        if self.model_type == ModelType.image_text:
            self.model_type_config['best_date'] = best_date
            self.model_type_config['best_epochs'] = best_epochs
        if self.model_type != ModelType.text_only:
            self.model_type_config['IMAGE_SIZE'] = IMAGE_SIZE
        if self.model_type != ModelType.image_only:
            self.model_type_config['TEXT_LENGTH'] = TEXT_LENTH
            self.model_type_config['EMBED_SIZES'] = EMBED_SIZES

        self.trainable = is_trainable
        self.is_primary_model = is_primary_model

        # Book-keeping directories
        self.train_log_dir = self.results_subdirectory + self.model_name.name + train_log_directory_suffix
        self.checkpoint_dir = self.results_subdirectory + self.model_name.name + checkpoint_directory_suffix
        self.test_log_dir = self.results_subdirectory + self.model_name.name + test_log_directory_suffix
        self.train_logfile_name = self.train_log_dir + "out.tsv"

        # Placeholders
        self.train_mode = tf.placeholder(dtype=tf.bool, name='train_mode_placeholder')
        self.labels_placeholder = tf.placeholder(dtype="float", shape=[None, NUM_CLASSES], name='labels_placeholder')
        self.dropout_keep_prob = tf.placeholder(dtype="float", name='dropout_keep_prob_placeholder')

        # Global step for optimizer
        self.global_step = None

        # Output probabilities, predictions tensors and loss scalar
        self.scores = None
        self.probabilities = None
        self.predictions = None
        self.loss = None

        # Optimization
        self.learning_rate = learning_rate_dict[self.model_type] * (batch_size_dict[self.model_type] / 256.)
        self.optimizer = None
        self.train_op = None

        # Accuracy stats
        self.accuracy = None
        self.sum_accuracy = None
        self.acc_summary = None
        self.loss_summary = None

        # Model save & restore, tensorboard
        self.saver = None
        self.global_variable_init = None
        self.train_summary_op = None
        self.train_writer = None

    def train(self, train_ids, train_labels, train_texts, val_ids, val_labels, val_texts, retrain_model_filename=None):
        """Train the model on given data, saving model checkpoints and log files.
        
        Parameters
        ----------
        :param train_ids: array-like, of shape [NUM_TRAIN_SAMPLES] and type int

        :param train_labels: array-like, of shape [NUM_TRAIN_SAMPLES, NUM_CLASSES] and type int

        :param train_texts: array-like, of shape [NUM_TRAIN_SAMPLES, TEXT_LENGTH] and type int

        :param val_ids: array-like, of shape [NUM_VAL_SAMPLES] and type int

        :param val_labels: array-like, of shape [NUM_VAL_SAMPLES, NUM_CLASSES] and type int

        :param val_texts: array-like, of shape [NUM_VAL_SAMPLES, TEXT_LENGTH] and type int

        :param retrain_model_filename: str, if None, model is trained from scratch,
            Else this param states where the ckpt file resides. The model is further trained from that ckpt.
        """
        if self.model_type != ModelType.text_only:
            iterator_train = get_dataset_iterator_from_tfrecords(data_type='train',
                                                                 batch_size=self.batch_size,
                                                                 model_type=self.model_type)
            iterator_val = get_dataset_iterator_from_tfrecords(data_type='validation',
                                                               batch_size=self.batch_size,
                                                               model_type=self.model_type)

            iterator_train_init = iterator_train.initializer
            iterator_val_init = iterator_val.initializer
            next_element_train = iterator_train.get_next()
            next_element_val = iterator_val.get_next()
        else:
            iterator_train_init, next_element_train, iterator_val_init, next_element_val = None, None, None, None

        vars_to_restore = [var for var in tf.global_variables()]
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
        """Run prediction on the test data by restoring a pre-trained model

        Parameters
        ----------
        :param date: str, folder name (timestamp) from which to retrieve pre-trained model

        :param num_epochs: int, the number of epochs for the pre-trained model

        :param test_ids: np.array of shape = [test_size], contain question ids in test set

        :param test_labels: np.array of shape = [test_size, NUM_CLASSES]

        :param test_texts: np.array of shape = [test_size, TEXT_LENGTH]
        """
        print("\nStarting to test %s model" % self.model_name.name)
        print("Test Data size: %d" % len(test_ids))

        # For image+text data, we use tf.dataset, while for text-only, we use simple iteration over loaded data.
        if self.model_type != ModelType.text_only:
            iterator_test = get_dataset_iterator_from_tfrecords(data_type='test',
                                                                batch_size=self.batch_size, model_type=self.model_type)
            iterator_test_init = iterator_test.initializer
            next_element_test = iterator_test.get_next()
        else:
            iterator_test, iterator_test_init, next_element_test = None, None, None

        vars_to_restore = tf.global_variables()
        saver = tf.train.Saver(vars_to_restore)

        # Make directory to write results to
        if not os.path.exists(self.test_log_dir):
            os.makedirs(self.test_log_dir)

        with tf.Session() as sess:
            self.restore_model(sess, date, num_epochs, saver)
            # Run the tests and get accuracy, predicted labels, confusion matrix etc.
            test_accuracy, labels, results, conf_matrix = self._run_tests(sess, test_ids, test_labels, test_texts,
                                                                          iterator_test_init, next_element_test)
            self._write_results_to_file(conf_matrix, date, labels, results, test_ids, test_accuracy)

    def retrain(self, date, num_epochs, train_ids, train_labels, train_texts, val_ids, val_labels, val_texts):
        """Load a pre-trained model, and start retraining it."""
        print("\nStarting to retrain %s model\n" % self.model_name.name)
        # This is the name under which the model usually gets stored
        model_filename = get_stored_checkpoint_filename(model_type=self.model_type,
                                                        model_name=self.model_name, date=date, num_epochs=num_epochs)
        # Start retraining
        self.train(train_ids, train_labels, train_texts, val_ids, val_labels, val_texts,
                   retrain_model_filename=model_filename)

    def _train_helper_SGD(self, sess, train_writer, train_ids, train_labels, train_texts,
                          val_ids, val_labels, val_texts, iterator_train_init, next_element_train, iterator_val_init,
                          next_element_val):
        """Abstract method"""
        raise NotImplementedError

    def _run_tests(self, sess, test_ids, test_labels, test_texts, iterator_test_init, next_element_test):
        """Abstract method"""
        raise NotImplementedError

    def _write_results_to_file(self, conf_matrix, date, labels, results, test_ids, test_accuracy):
        """Write two files: test_logits.tsv and test_result.tsv.

        First one will contain the predictions for each sample,

        Second contains confusion matrix for the categories and final accuracy.
        """
        print('\nWriting results to file ...')

        with open(self.test_log_dir + 'test_logits.tsv', 'w') as handle1, \
                open(self.results_subdirectory + self.model_name.name + '/train/' + date + '/test_result.tsv',
                     'w') as handle2:
            if conf_matrix is not None:
                for row in conf_matrix:
                    r = ''
                    for col in row:
                        r = r + str(col).rjust(4) + ','
                    handle2.write(r + '\n')
            for i in xrange(len(labels)):
                o = '\t'.join([str(test_ids[i]), str(labels[i]), str(results[i])])
                handle1.write(o + '\n')
            handle2.write("test accuracy = %g" % test_accuracy)

        print('Done writing results to file\n')
        print('\n***Test Accuracy = %g***\n' % test_accuracy)

    def _set_predictions_optimizer_and_loss(self):
        if USE_MULTILABEL:
            self.probabilities = tf.nn.sigmoid(self.scores, name="probabilities")
            self.predictions = tf.round(self.probabilities, name="predictions")
        else:
            self.probabilities = tf.nn.softmax(self.scores, name="probabilities")
            self.predictions = tf.argmax(self.probabilities, 1, name="predictions")

        if self.trainable:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        with tf.variable_scope("loss"):
            if USE_MULTILABEL:
                # Calculate sigmoid cross-entropy loss
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels_placeholder,
                                                                                   logits=self.scores))
            else:
                # Calculate mean cross-entropy loss
                self.loss = -tf.reduce_mean(self.labels_placeholder * tf.log(tf.clip_by_value(self.probabilities,
                                                                                              1e-10, 1.0)))

        if decay_learning_rate:
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                            DECAY_STEP_LEARNING_RATE, DECAY_RATE_LEARNING_RATE,
                                                            staircase=True)
        elif scale_learning_rate:
            initial_learning_rate = self.learning_rate
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

        if self.model_type != ModelType.image_only:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_name.name)
        else:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

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

        # Calculate Accuracy
        with tf.variable_scope("accuracy"):
            if not USE_MULTILABEL:
                self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels_placeholder, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
                self.sum_accuracy = tf.reduce_sum(tf.cast(self.correct_predictions, "float"), name="sum_accuracy")
            else:
                self.correct_predictions = tf.reduce_all(tf.equal(self.predictions, self.labels_placeholder), axis=1)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name='accuracy')
                self.sum_accuracy = tf.reduce_sum(tf.cast(self.correct_predictions, "float"), name="sum_accuracy")

        # Summaries for loss and accuracy
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        vars_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(var_list=vars_to_restore, max_to_keep=None)

        # Train Summaries at Tensorboard
        self.train_summary_op = tf.summary.merge([self.loss_summary, self.acc_summary])

        # Define the file writer for tensorboard
        if self.is_primary_model:
            self.train_writer = tf.summary.FileWriter(self.train_log_dir + "train_tensorboard", tf.get_default_graph())

        print('Completed %s object construction.' % self.model_name.name)

    def restore_mid_training(self, sess, num_epochs):
        """Restore a pre-trained model while training - specifically when validation accuracy has decreased,
            and we retore the best model so far, and run training with reduced learning rate.

        Parameters
        ----------
        :param sess: tf.Session() object

        :param num_epochs: int, the best model's epoch number
        """
        vars_to_restore = [var for var in tf.global_variables() if 'learning_rate_var' not in var.name]
        saver = tf.train.Saver(vars_to_restore)
        self.global_variable_init = tf.global_variables_initializer()
        model_filename = get_stored_checkpoint_filename(self.model_type, self.model_name, TIME_STAMP, num_epochs)
        self.restore_model_from_filename(sess, model_filename=model_filename, saver=saver)

    def restore_model(self, sess, date, num_epochs, saver=None):
        """Instantiate a tf.train.Saver object and restore saved model at model_filename through it

        Parameters
        ----------
        :param sess: tensorflow.Session object, current open session

        :param date: str, folder name timestamp where the model is stored

        :param num_epochs: int or str, the epoch number from which model is to be restored

        :param saver: tf.train.Saver() object
        """
        # This is the name under which the model usually gets stored
        model_filename = get_stored_checkpoint_filename(model_type=self.model_type,
                                                        model_name=self.model_name, date=date, num_epochs=num_epochs)
        self.restore_model_from_filename(sess, model_filename=model_filename, saver=saver)

    @staticmethod
    def restore_model_from_filename(sess, model_filename, saver=None):
        """Instantiate a tf.train.Saver object and restore saved model at model_filename through it

        Parameters
        ----------
        :param sess: tensorflow.Session object, current open session

        :param model_filename: str, the name under which the model usually gets stored

        :param saver: tf.train.Saver() object

        """
        print("\nRestoring model from %s ..." % model_filename)
        if saver is None:
            vars_to_restore = tf.global_variables()
            saver = tf.train.Saver(vars_to_restore)

        saver.restore(sess, model_filename)
        print("Done restoring model.")

    def _train_setup(self, train_ids, val_ids, batch_size):
        """Setup training by printing preliminary information, making reqd directories, initializing tf session,
        and making required log file. Return the tf.Session object and opened log file handle.

        P.S: Please remember to close the returned sess handle, the code structure was not apt for a "with" clause.

        Parameters
        ----------
        :param train_ids: np.array of ints

        :param val_ids: np.array of ints

        :param batch_size: int

        Returns
        -------
        sess: [tf.Session() object, file_handle]
        """
        # Make directory to store checkpoints.
        os.makedirs(self.checkpoint_dir)

        self._write_config_file()

        print('\nStarting to train %s model' % self.model_name.name)

        # Print stats
        print("Total train samples: %d" % len(train_ids))
        print("Batch Size: %d, Num batches: %d" % (batch_size, int(len(train_ids) / batch_size)))
        print("Total validation samples: %d" % len(val_ids))

        # Session: launch the graph
        sess = tf.Session()
        # Initialize all variables
        print("Running global variable init")
        sess.run(self.global_variable_init)
        print("Done running global variable init")
        return sess

    def validation_test(self, date, num_epochs, val_ids, val_labels, val_texts):
        if self.model_type != ModelType.text_only:
            iterator_validation = get_dataset_iterator_from_tfrecords(data_type='validation',
                                                                      batch_size=self.batch_size,
                                                                      model_type=self.model_type)
            iterator_validation_init = iterator_validation.initializer
            next_element_val = iterator_validation.get_next()
        else:
            iterator_validation_init, next_element_val = None, None

        vars_to_restore = tf.global_variables()
        saver = tf.train.Saver(vars_to_restore)
        self.global_variable_init = tf.global_variables_initializer()
        with tf.Session() as sess:
            self.restore_model(sess, date, num_epochs, saver)
            # self.restore_base_models(sess)
            # tf.get_default_graph().finalize()
            self._run_validation_test(sess, val_ids, val_labels, val_texts, iterator_validation_init, next_element_val)

    def _run_validation_test(self, sess, val_ids, val_labels, val_texts, iterator_validation, next_element_val):
        pass

    def _write_config_file(self):
        """Write the configuration file for the model"""
        with open(self.checkpoint_dir + "config", "w") as handle:
            for key, value in global_config.iteritems():
                handle.write("%-25s %-s\n" % (key, value))
            if self.config is not None:
                handle.write("\n#----------------------------- MODEL CONFIG -----------------------------#\n")
                for key, value in self.config.iteritems():
                    handle.write("%-25s %-s\n" % (key, value))
            handle.write("\n#--------------------------- MODEL TYPE CONFIG ---------------------------#\n")
            for key, value in self.model_type_config.iteritems():
                handle.write("%-25s %-s\n" % (key, value))

    def restore_base_models(self, sess):
        # TODO: Make abstact and implement for all models
        pass

    # noinspection PyUnresolvedReferences
    def early_stopping_procedure(self, sess, epochs_with_current_lrate, val_acc_history, min_increment,
                                 tolerable_decrease):
        """Implement procedure for decaying learning rate acc to validation accuracy history

        Parameters
        ----------
        :param sess: tf.Session() object

        :param epochs_with_current_lrate: int, number of epochs completed with current learning rate

        :param val_acc_history: list of floats, containing validation accuracies of all completed epochs

        :param min_increment: float, minimum increment over validation accuracy viewed as significant

        :param tolerable_decrease: float, tolerable amount of decrease in val acc, otherwise learning rate is tanked.

        Returns
        -------
        :return: bool, whether the model should early stop on account of no val acc increase, or not.
        """
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
            elif epochs_with_current_lrate >= 1 and (max(val_acc_history) - val_acc_history[-1]) > tolerable_decrease:
                best_epoch = np.argmax(val_acc_history)
                self.restore_mid_training(sess=sess, num_epochs=best_epoch)
                self.learning_rate.assign(self.learning_rate / 2.).eval(session=sess)

        # early_stop_bool=True for early stopping if we have more than 15 epochs,
        # and less than min_increment over last ten epochs
        early_stop_bool = (len(val_acc_history) > 15) and (val_acc_history[-1] < (val_acc_history[-10] + min_increment))
        return early_stop_bool, epochs_with_current_lrate
