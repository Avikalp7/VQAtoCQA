# -*- coding: utf-8 -*-

"""
The user interacts via this module.
This module is responsible for
    * Parsing command line arguments
    * Issuing construction of apt classifier with provided model parameters
    * Issuing train/test/val for the classifier

TODO: Replace printing with logging

Standard command:
$ python run.py -model text_cnn -train 1

Showing command with all options (excluding date & epochs which are not required when train=1):

* To first train the model, and on end of training run it on test:
$ python run.py -model image_cnn -train 1 -test 1

* To restore the model stored in 18Y05M25D13h23m, from its 15th epoch checkpoint, and run it on test set:
$ python run.py -model text_cnn -test 1 -date 18Y05M25D13h23m -nepochs 15
"""
# Copyright (C) 2018 Yahoo Japan Corporation (Licensed under CC BY-NC-SA 4.0)
# https://creativecommons.org/licenses/by-nc-sa/4.0/


import argparse

import tensorflow as tf

import image_models
import text_models
import image_text_models
from utils import load_train_val_data, load_test_data, load_val_data
from global_hyperparams import best_date, best_epochs, ModelType, ModelName, \
    update_learning_rate_dict, update_training_epochs_dict


def parse_command_line_args():
    """ Parse command line arguments. All are optional, except that if test=1 & train=0,
    then date and nepochs are required.

    :return: list [str, bool, bool, str, str] containing model name, bools for whether to train & test, date string,
        and number of epochs as string.
    :raises ValueError: if test
    """
    parser = argparse.ArgumentParser(description='Use python <filename.py> -model <image_model_name here> '
                                                 '-train <0/1 here> -test <0/1 here> -date <date here> '
                                                 '-nepochs <int here>')
    parser.add_argument('-model', help='Name of model: only %s allowed for now. '
                                       'Default is "%s"' % (str([mn.name for mn in list(ModelName)]),
                                                            ModelName.text_cnn.name),
                        type=str, required=False, default=ModelName.text_cnn.name)
    parser.add_argument('-train', help='Boolean 0/1, whether to train from scratch. Default is 0',
                        type=int, required=False, default=0)
    parser.add_argument('-test', help='Boolean 0/1, whether to test the model. Default is 0',
                        type=int, required=False, default=0)
    parser.add_argument('-val', help='Boolean 0/1, whether to validate the model. Default is 0',
                        type=int, required=False, default=0)
    parser.add_argument('-retrain', help='Boolean 0/1, whether to retrain the model. Default is 0',
                        type=int, required=False, default=0)
    parser.add_argument('-date', help='date stamp of the folder storing the saved model',
                        type=str, required=False, default=None)
    parser.add_argument('-nepochs', help='number of epochs the trained model was ran',
                        type=int, required=False, default=None)
    parser.add_argument('-lrate', help='learning rate for the model, default is in global_hyperparams',
                        type=float, required=False, default=None)
    parser.add_argument('-tr_epochs', help='number of training epochs for model, default is in global_hyperparams',
                        type=int, required=False, default=None)
    parser.add_argument('-nlayers', help='Number of stacked att layers for att models',
                        type=int, required=False, default=1)

    args = vars(parser.parse_args())
    model_name = ModelName[args['model']]
    train_flag = bool(args['train'])
    test_flag = bool(args['test'])
    val_flag = bool(args['val'])
    retrain_flag = bool(args['retrain'])
    date = args['date']
    num_epochs = args['nepochs']
    learning_rate = args['lrate']
    training_epochs = args['tr_epochs']
    num_att_layers = args['nlayers']

    if (val_flag or test_flag) and (date is None):
        print("WARNING: Date & Epochs not mentioned, loading from parameters mentioned in global_hyperparams.py file")
        date = best_date[model_name]
        num_epochs = best_epochs[model_name]

    return model_name, train_flag, test_flag, val_flag, retrain_flag, date, num_epochs, \
        learning_rate, training_epochs, num_att_layers


def train_model(classifier):
    """ Train the given classifier model on the training data present in data_directory.

    :param classifier: Object with class derived from BaseModel
    :return: None
    """
    train_ids, train_labels, train_texts, val_ids, val_labels, val_texts = \
        load_train_val_data(load_texts=(classifier.model_type != ModelType.image_only))
    classifier.train(train_ids, train_labels, train_texts, val_ids, val_labels, val_texts)


def test_model(classifier, date, num_epochs):
    """ Restore the classifier from given data and number of epochs, and run on test set

    :param classifier: Object with class derived from BaseModel
    :param date: str, timestamp that is the name of the folder containing classifier's data
    :param num_epochs: int, the epoch number from which checkpoint is to be retrieved.
    :return: None
    """
    test_labels, test_ids, test_texts = load_test_data(load_texts=(classifier.model_type != ModelType.image_only))
    classifier.test(date, num_epochs, test_ids, test_labels, test_texts)


def validate_model(classifier, date, num_epochs):
    """ Restore the classifier from given data and number of epochs, and run on validation set

    :param classifier: Object with class derived from BaseModel
    :param date: str, timestamp that is the name of the folder containing classifier's data
    :param num_epochs: int, the epoch number from which checkpoint is to be retrieved.
    :return: None
    """
    val_labels, val_ids, val_texts = load_val_data(load_texts=(classifier.model_type != ModelType.image_only))
    classifier.validation_test(date, num_epochs, val_ids, val_labels, val_texts)


def retrain_model(classifier, date, num_epochs):
    """ Restore the classifier from given data and number of epochs, and start retraining

    :param classifier: Object with class derived from BaseModel
    :param date: str, timestamp that is the name of the folder containing classifier's data
    :param num_epochs: int, the epoch number from which checkpoint is to be retrieved.
    :return: None
    """
    load_texts = not (classifier.model_type == ModelType.image_only)
    train_ids, train_labels, train_texts, val_ids, val_labels, val_texts = load_train_val_data(load_texts)
    classifier.retrain(date, num_epochs, train_ids, train_labels, train_texts, val_ids, val_labels, val_texts)


def get_classifier_model(model_name, num_att_layers):
    """ Given the model name, construct the corresponding BaseModel derived object and return it

    :param model_name: ModelName enum (from global_hyperparams)
    :param num_att_layers: int or None, number of attention layers to be used for SAN
    :return: BaseModel derived class object
    :raises: ValueError, if model_name in not a implemented classifier model
    """
    # Text Only Models
    if model_name == ModelName.text_cnn:
        classifier = text_models.TextCNN(filter_sizes=(1, 2, 3), is_trainable=True)
    elif model_name == ModelName.hie_text:
        classifier = text_models.HieText(is_trainable=True, is_primary_model=True)

    # Image Only Models
    elif model_name == ModelName.image_cnn:
        classifier = image_models.ImageCNN(is_trainable=True)
    elif model_name == ModelName.image_cnn_v2:
        classifier = image_models.ImageCNNV2(is_trainable=True)
    elif model_name == ModelName.resnet:
        classifier = image_models.Resnet(is_trainable=True, train_last_block=False)
    elif model_name == ModelName.resnet_clf:
        classifier = image_models.ResnetClf(is_trainable=True)

    # Image Text Models
    elif model_name == ModelName.embedding_concat_semifreeze:
        classifier = image_text_models.EmbeddingConcatWithSemiFreeze(is_trainable=True)
    elif model_name == ModelName.stacked_attention_with_semi_freeze_cnn:
        classifier = image_text_models.StackedAttentionWithSemiFreezeCNN(nlayers=num_att_layers,
                                                                         is_trainable=True)
    elif model_name == ModelName.hie_co_att:
        classifier = image_text_models.HieCoAtt(is_primary_model=True, is_trainable=True)

    # Image-Text Novel Model
    elif model_name == ModelName.aux_task_model:
        classifier = image_text_models.AuxTaskModel()

    else:
        raise ValueError("Invalid model name=%s provided" % model_name)

    return classifier


def main():

    # Get the parameter value from command line arguments
    model_name, train_flag, test_flag, val_flag, retrain_flag, \
        date, num_epochs, learning_rate,\
        training_epochs, num_att_layers = parse_command_line_args()

    with tf.Graph().as_default():
        classifier = get_classifier_model(model_name, num_att_layers)

        # Override values of learning rate and training epochs if explicitly specified by the user
        if learning_rate is not None:
            update_learning_rate_dict(classifier.model_type, learning_rate)
        if training_epochs is not None:
            update_training_epochs_dict(classifier.model_type, training_epochs)

        # Carry out requested action using the classifier
        if train_flag:
            train_model(classifier)
        if test_flag:
            test_model(classifier, date, num_epochs)
        if val_flag:
            validate_model(classifier, date, num_epochs)
        if retrain_flag:
            retrain_model(classifier, date, num_epochs)


if __name__ == '__main__':
    main()
