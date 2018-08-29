# -*- coding: utf-8 -*-
"""Module providing common utility functions

TODO: Add more documentation
"""
# Copyright (C) 2018 Yahoo Japan Corporation (Licensed under CC BY-NC-SA 4.0)
# https://creativecommons.org/licenses/by-nc-sa/4.0/


import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image

from paths import data_directory
from global_hyperparams import NUM_CLASSES, IMAGE_SIZE, TEXT_LENTH, USE_TRUNC_VOCAB, \
    USE_MULTILABEL, USE_MERGED_LABELS, USE_2014_DATA

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def load_train_val_data(load_texts=True):
    """Read files from data_directory and return np arrays for train labels and ids & validation labels & ids.

    Parameters
    ----------
    :param load_texts: bool, whether to load text data, or just return None for it

    Returns
    -------
    train_ids: array-like, of shape [NUM_TRAIN_QUESTIONS] containing Chiebukuro question ids

    train_labels: array-like, of shape [NUM_TRAIN_QUESTIONS, NUM_CLASSES]

    train_texts: array-like, of shape [NUM_TRAIN_QUESTIONS, TEXT_LENGTH]

    val_ids: array-like, of shape [NUM_VAL_QUESTIONS] containing Chiebukuro question ids

    val_labels: array-like, of shape [NUM_VAL_QUESTIONS, NUM_CLASSES]

    val_texts: array-like, of shape [NUM_VAL_QUESTIONS, TEXT_LENGTH]
    """
    print("\nLoading training & validation data ...")
    train_ids = np.load(data_directory + 'train_ids%s.npy' % ('_2014' if USE_2014_DATA else ''))
    print("Loaded train ids")
    val_ids = np.load(data_directory + 'val_ids%s.npy' % ('_2014' if USE_2014_DATA else ''))
    print("Loaded val ids")

    train_labels = np.load(data_directory + 'train_label_ext%d%s%s%s.npy'
                           % (NUM_CLASSES, '_2014' if USE_2014_DATA else '', '_ml' if USE_MULTILABEL else '',
                              '_merged' if USE_MERGED_LABELS else ''))
    val_labels = np.load(data_directory + 'val_label_ext%d%s%s%s.npy'
                         % (NUM_CLASSES, '_2014' if USE_2014_DATA else '', '_ml' if USE_MULTILABEL else '',
                            '_merged' if USE_MERGED_LABELS else ''))
    print("Done loading labels")

    if TEXT_LENTH == 155:
        train_texts = np.load(data_directory + 'train155.npy') if load_texts else None
        val_texts = np.load(data_directory + 'val155.npy') if load_texts else None
    elif TEXT_LENTH == 160:
        if USE_TRUNC_VOCAB:
            train_texts = np.load(data_directory + 'train160_trunc%s.npy' % ('_2014' if USE_2014_DATA else '')) \
                if load_texts else None
            val_texts = np.load(data_directory + 'val160_trunc%s.npy' % ('_2014' if USE_2014_DATA else '')) \
                if load_texts else None
        else:
            train_texts = np.load(data_directory + 'train160%s.npy' % ('_2014' if USE_2014_DATA else '')) \
                if load_texts else None
            val_texts = np.load(data_directory + 'val160%s.npy' % ('_2014' if USE_2014_DATA else '')) \
                if load_texts else None
    else:
        train_texts = resize_input(np.load(data_directory + 'train.npy'), TEXT_LENTH) if load_texts else None
        val_texts = resize_input(np.load(data_directory + 'val.npy'), TEXT_LENTH) if load_texts else None

    print("Done loading training & validation data.")
    return train_ids, train_labels, train_texts, val_ids, val_labels, val_texts


def load_test_data(load_texts=True):
    """Read files from data_directory and return np arrays for test labels and ids.

    Parameters
    ----------
    :param load_texts: bool, whether to load text data, or just return None for it
    """
    print("\nLoading test data ...")
    test_labels = np.load(data_directory + 'test_label_ext%d%s%s%s.npy'
                          % (NUM_CLASSES, '_2014' if USE_2014_DATA else '', '_ml' if USE_MULTILABEL else '',
                             '_merged' if USE_MERGED_LABELS else ''))

    test_ids = np.load(data_directory + 'test_ids%s.npy' % ('_2014' if USE_2014_DATA else ''))

    test_texts = None
    if load_texts:
        if TEXT_LENTH == 155:
            test_texts = np.load(data_directory + 'test155.npy')
        elif TEXT_LENTH == 160:
            if USE_TRUNC_VOCAB:
                test_texts = np.load(data_directory + 'test160_trunc%s.npy' % ('_2014' if USE_2014_DATA else ''))
            else:
                test_texts = np.load(data_directory + 'test160%s.npy' % ('_2014' if USE_2014_DATA else ''))
        else:
            test_texts = resize_input(np.load(data_directory + 'test%s.npy' % ('_2014' if USE_2014_DATA else '')),
                                      TEXT_LENTH)

    print("Done loading test data.")
    return test_labels, test_ids, test_texts


def load_val_data(load_texts=True):
    """Read files from data_directory and return np arrays for test labels and ids.

    Parameters
    ----------
    :param load_texts: bool, whether to load text data, or just return None for it
    """
    print("\nLoading val data ...")
    val_labels = np.load(data_directory + 'val_label_ext%d%s%s%s.npy'
                         % (NUM_CLASSES, '_2014' if USE_2014_DATA else '', '_ml' if USE_MULTILABEL else '',
                            '_merged' if USE_MERGED_LABELS else ''))
    print("Done loading val labels")

    val_ids = np.load(data_directory + 'val_ids%s.npy' % ('_2014' if USE_2014_DATA else ''))
    print("Done loading val ids")

    val_texts = None
    if load_texts:
        if TEXT_LENTH == 155:
            val_texts = np.load(data_directory + 'val155.npy')
        elif TEXT_LENTH == 160:
            if USE_TRUNC_VOCAB:
                val_texts = np.load(data_directory + 'val160_trunc%s.npy' % ('_2014' if USE_2014_DATA else ''))
            else:
                val_texts = np.load(data_directory + 'val160%s.npy' % ('_2014' if USE_2014_DATA else ''))
        else:
            val_texts = resize_input(np.load(data_directory + 'val%s.npy' % ('_2014' if USE_2014_DATA else '')),
                                     TEXT_LENTH)

    print("Done loading val data.")
    return val_labels, val_ids, val_texts


def right_align(sequence):
    right_aligned_seq = np.zeros(np.shape(sequence))
    lengths = np.argmax(sequence == 0, axis=1)

    lengths[np.where(lengths == 0)] = TEXT_LENTH

    cols = np.shape(sequence)[1]
    for row in range(np.shape(sequence)[0]):
        right_aligned_seq[row][cols - lengths[row]:cols] = sequence[row][0:lengths[row]]
    return right_aligned_seq


def _train_image_preprocess(image, label_tensor, arr_idx):
    seed = 12345
    image = tf.image.random_flip_left_right(image, seed=seed)

    if np.random.randint(2) == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255., seed=seed)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.image.random_hue(image, max_delta=0.032, seed=seed)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5, seed=seed)
    else:
        image = tf.image.random_brightness(image, max_delta=32. / 255., seed=seed)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.image.random_hue(image, max_delta=0.032, seed=seed)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    # Convert to [-1, 1]
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return _make_image_tensor(image, label_tensor, arr_idx)


def _eval_image_preprocess(image, label_tensor, arr_idx):
    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    # Convert to [-1, 1]
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return _make_image_tensor(image, label_tensor, arr_idx)


def _vis_image_preprocess(image, label_tensor, arr_idx):
    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return _make_image_tensor(image, label_tensor, arr_idx)


def _make_image_tensor(image, label_tensor, arr_idx):
    return tf.reshape(image, shape=[IMAGE_SIZE*IMAGE_SIZE*3]), label_tensor, arr_idx


def _parse(example_proto):
    """Parses the tfrecord files, returns the image encoding string, the label as int, and idx name of sample"""
    features = {
        'image/arr_idx': tf.FixedLenFeature([], tf.int64),
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
    }

    if not USE_2014_DATA:
        features['image/class/label'] = tf.FixedLenFeature([], tf.int64)

    parsed_features = tf.parse_single_example(example_proto, features)

    arr_idx = tf.cast(parsed_features['image/arr_idx'], dtype=tf.int32)
    if USE_2014_DATA:
        label = tf.cast(0, dtype=tf.int32)
    else:
        label = tf.cast(parsed_features['image/class/label'], dtype=tf.int32)
    encoded_image = parsed_features["image/encoded"]
    image_scaled, label_tensor, arr_idx = _parse_function(encoded_image, label, arr_idx)
    return image_scaled, label_tensor, arr_idx


def _parse_function(encoded_image, label, arr_idx):
    image_decoded = tf.image.decode_jpeg(encoded_image, channels=3)
    # This will convert to float values in [0, 1]
    image_scaled = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_scaled = tf.image.resize_images(image_scaled, [IMAGE_SIZE, IMAGE_SIZE])
    label_tensor = tf.one_hot(indices=label, depth=NUM_CLASSES, dtype=tf.int32, name='label_tensor')
    return image_scaled, label_tensor, arr_idx


def get_dataset_iterator_from_tfrecords(data_type, model_type, batch_size, vis=False):
    # ignore unused arg. TODO: Remove model_type arg
    _ = model_type
    train_files_num = 1024 if not USE_2014_DATA else 256
    val_test_files_num = 128 if not USE_2014_DATA else 32
    if data_type == 'validation' or data_type == 'test':
        filenames = ["../tf_data_with_idx%s/%s-%05d-of-00%s%d"
                     % ('_2014' if USE_2014_DATA else '',
                        data_type, itr, '0' if USE_2014_DATA else '',
                        val_test_files_num) for itr in range(val_test_files_num)]
    elif data_type == 'train':
        filenames = ["../tf_data_with_idx%s/%s-%05d-of-0%s%d"
                     % ('_2014' if USE_2014_DATA else '',
                        data_type, itr, '0' if USE_2014_DATA else '',
                        train_files_num) for itr in range(train_files_num)]
    else:
        raise ValueError("Incorrect data_type=%s in parse_tfrecrods" % data_type)

    dataset = tf.contrib.data.TFRecordDataset(filenames)

    if USE_2014_DATA:
        buffer_size = 3000
    else:
        buffer_size = 10000

    if data_type == 'train':
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=12345)

    dataset = dataset.map(_parse, num_threads=batch_size, output_buffer_size=batch_size*2)
    if data_type == 'train':
        dataset = dataset.map(_train_image_preprocess, num_threads=batch_size, output_buffer_size=batch_size*2)
    else:
        if not vis:
            dataset = dataset.map(_eval_image_preprocess, num_threads=batch_size, output_buffer_size=batch_size*2)
        else:
            dataset = dataset.map(_vis_image_preprocess, num_threads=batch_size, output_buffer_size=batch_size*2)

    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()

    return iterator


def get_dataset_iterator_from_tfrecords_answerer(data_type, batch_size):
    train_files_num = 128
    val_test_files_num = 16
    if data_type == 'validation' or data_type == 'test':
        filenames = ["../tf_data_with_idx_answerer/%s-%05d-of-000%d"
                     % (data_type, itr, val_test_files_num) for itr in range(val_test_files_num)]
    elif data_type == 'train':
        filenames = ["../tf_data_with_idx_answerer/%s-%05d-of-00%d"
                     % (data_type, itr, train_files_num) for itr in range(train_files_num)]
    else:
        raise ValueError("Incorrect data_type=%s in parse_tfrecrods" % data_type)

    dataset = tf.contrib.data.TFRecordDataset(filenames)

    buffer_size = 1500

    if data_type == 'train':
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=12345)

    dataset = dataset.map(_parse, num_threads=batch_size, output_buffer_size=batch_size*2)

    dataset = dataset.map(_eval_image_preprocess, num_threads=batch_size, output_buffer_size=batch_size * 2)

    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()

    return iterator


def get_coco_dataset_iterator_from_tfrecords(batch_size):
    files_num = 8
    filenames = ["../tf_data_with_idx_coco2/coco-0000%d-of-0000%d" % (itr, files_num) for itr in range(files_num)]

    dataset = tf.contrib.data.TFRecordDataset(filenames)

    dataset = dataset.map(_parse, num_threads=batch_size, output_buffer_size=batch_size*2)
    dataset = dataset.map(_eval_image_preprocess, num_threads=batch_size, output_buffer_size=batch_size * 2)

    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()

    return iterator


def longest(input_list):
    return max(len(element) for element in input_list)


def resize_input(np_array, size):
    return np_array[:, :size]


def text_batch_generator(batch_size, text_data, label_data):
    indices = np.random.permutation(range(len(text_data)))
    for i in xrange(0, len(text_data), batch_size):
        ind = indices[i:i+batch_size]
        batch_text = []
        batch_labels = []
        for j in ind:
            batch_text.append(text_data[j])
            batch_labels.append(label_data[j])
        yield np.asarray(batch_text), np.asarray(batch_labels)


def text_batch_from_ids(ids, text_data):
    return np.asarray(text_data[ids])


def batch_norm(x, n_out, phase_train):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return normed


def mono_init(img, img_size):
    new_img = np.zeros((img_size, img_size, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img


def la_init(img, img_size):
    new_img = np.zeros((img_size, img_size, 3))
    for i in range(3):
        new_img[:, :, i] = img[:, :, 0]
    return new_img


def image_reader(data_dir, name_list, image_size):
    Image.MAX_IMAGE_PIXELS = None
    return_image = [None]*len(name_list)
    for i in range(len(name_list)):
        file_name = str(name_list[i]) + ".jpg"
        try:
            img = np.array(Image.open(data_dir + file_name, "r").resize((image_size, image_size)), dtype=np.float32)
        except AttributeError as e:
            print("Encountered exception while reading images:", e)
            sys.exit(-1)

        img = np.array(img / 255.0)
        if len(img.shape) != 3:
            img = mono_init(img, image_size)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        elif img.shape[2] == 2:
            img = la_init(img, image_size)
        img = img.reshape(image_size*image_size*3)
        return_image[i] = img
    return_image = np.asarray(return_image)
    return return_image


def image_batch_generator(images_directory, batch_size, name_data, label_data, image_size=128):
    indices = np.random.permutation(range(len(name_data)))
    for i in range(0, len(name_data), batch_size):
        ind = indices[i:i+batch_size]
        batch_name = []
        batch_labels = []
        for j in ind:
            batch_name.append(name_data[j])
            batch_labels.append(label_data[j])
        batch_image = image_reader(images_directory,
                                   batch_name, image_size)
        yield np.asarray(batch_image), np.asarray(batch_labels), ind


def update_train_batch(global_step, S, loss_val, acc_val, train_writer, total_epoch_acc_val,
                       batches_completed_this_epoch):
    # S: train_summary
    total_epoch_acc_val += acc_val
    if global_step % 20 == 0:
        _print_batch_stats(global_step, loss_val, acc_val)
        if global_step % 100 == 0:
            print("Training accuracy till mini-batch %d: %0.4f" %
                  (batches_completed_this_epoch, total_epoch_acc_val/float(batches_completed_this_epoch)))
    train_writer.add_summary(S, global_step)
    global_step += 1
    return global_step, total_epoch_acc_val


def update_val_batch(num_steps, val_acc, val_acc_sum, total_batches):
    val_acc_sum += val_acc
    num_steps += 1
    if num_steps % 50 == 0:
        print("Completed %d validation batches of %d batches" % (num_steps, total_batches))
    return val_acc_sum, num_steps


def update_test_batch(batch_num, pred, mat, test_acc, conf_matrix, prediction, test_accuracy, total_batches):
    if batch_num == 0:
        new_conf_matrix = mat
        new_prediction = pred
    else:
        if batch_num % 50 == 0:
            print("Completed %d test batches of %d batches" % (batch_num, total_batches))
        new_conf_matrix = conf_matrix + mat
        new_prediction = np.vstack((prediction, pred))

    new_test_accuracy = test_accuracy + test_acc

    return new_conf_matrix, new_prediction, new_test_accuracy


def training_epoch_finish_routine(sess, val_acc_sum, num_samples, train_logfile_name, checkpoint_dir, epoch, saver):
    val_acc = val_acc_sum / float(num_samples)
    with open(train_logfile_name, "a+") as handle:
        handle.write("validation\t" + str(val_acc) + "\n")
        handle.flush()

    os.mkdir(checkpoint_dir + str(epoch) + 'epoch')
    _ = saver.save(sess, checkpoint_dir + str(epoch) + 'epoch/model.ckpt')

    print("\nCompleted training epoch number %d with Validation Accuracy %0.4f" % (epoch + 1, val_acc))


def test_finish_routine(test_labels, prediction):
    print("\nCompleted test data evaluation.")
    labels = np.argmax(test_labels, 1)
    results = np.argmax(prediction, 1)
    return labels, results


def _print_batch_stats(global_step, loss_val, acc_val):
    time_str = datetime.now().strftime('%y%m%d%H%M')
    print("{}: step {:5d}, loss {:8.6f}, acc {:6.4f}".format(time_str, global_step, loss_val, acc_val))
