"""
The user interacts via this module for answerer retrieval.

Standard command:
python run_ans.py -bmodel embedding_concat_semifreeze -train 1

"""

import argparse

import tensorflow as tf
import numpy as np

import answerer
from paths import data_directory
# from utils import load_train_val_data, load_test_data, load_val_data
# from paths import TIME_STAMP, get_bottleneck_data_subdirectory
from global_hyperparams import ModelName


def train_model(classifier):
    ans_train_labels = np.load(data_directory + 'answerer_train_labels.npy')
    ans_val_labels = np.load(data_directory + 'answerer_val_labels.npy')
    # ans_test_labels = np.load(data_directory + 'answerer_test_labels.npy')
    ans_train_texts = np.load(data_directory + 'answerer_train_texts.npy')
    ans_val_texts = np.load(data_directory + 'answerer_val_texts.npy')
    # ans_test_texts = np.load(data_directory + 'answerer_test_texts.npy')
    ans_train_ids = np.load(data_directory + 'answerer_train_ids.npy')
    ans_val_ids = np.load(data_directory + 'answerer_val_ids.npy')
    # ans_test_ids = np.load(data_directory + 'answerer_test_ids.npy')
    classifier.train(ans_train_ids, ans_train_labels, ans_train_texts, ans_val_ids, ans_val_labels, ans_val_texts)


# noinspection PyTypeChecker
def parse_command_line_args():
    """ Parse command line arguments. All are optional, except that if test=1 & train=0,
    then date and nepochs are required.

    :return: list [str, bool, bool, str, str] containing model name, bools for whether to train & test, date string,
        and number of epochs as string.
    """
    parser = argparse.ArgumentParser(description='Use python <filename.py> -model <image_model_name here> '
                                                 '-train <0/1 here> -test <0/1 here> -date <date here> '
                                                 '-nepochs <int here>')
    parser.add_argument('-bmodel', help='Name of base model: only %s allowed for now. '
                                        'Default is "%s"' % (str([mn.name for mn in list(ModelName)]),
                                                             ModelName.stacked_attention_with_semi_freeze_cnn),
                        type=str, required=False, default=ModelName.stacked_attention_with_semi_freeze_cnn.name)
    parser.add_argument('-train', help='Boolean 0/1, whether to train from scratch. Default is 0',
                        type=int, required=False, default=0)
    parser.add_argument('-test', help='Boolean 0/1, whether to test the model. Default is 0',
                        type=int, required=False, default=0)
    parser.add_argument('-val', help='Boolean 0/1, whether to validate the model. Default is 0',
                        type=int, required=False, default=0)

    args = vars(parser.parse_args())
    base_model_name = ModelName[args['bmodel']]
    train_flag = bool(args['train'])
    test_flag = bool(args['test'])
    val_flag = bool(args['val'])
    # retrain_flag = bool(args['retrain'])
    # vis_flag = bool(args['vis'])
    # bottleneck_flag = bool(args['bottleneck'])
    # date = args['date']
    # num_epochs = args['nepochs']
    # learning_rate = args['lrate']
    # training_epochs = args['tr_epochs']
    # decay_learning_rate = bool(args['decay'])
    # img_size = args['imgsize']
    # num_att_layers = args['nlayers']
    # dimk_div = args['dimk_div']

    # if test_flag and not train_flag:
    #     if date is None or num_epochs is None:
    #         raise ValueError("If testing mode is on & training mode is off, date and nepochs need to be provided.")
    # elif test_flag and train_flag:
    #     date = TIME_STAMP
    #     num_epochs = str(training_epochs_dict[ModelType.image_only] - 1)
    #
    # if decay_learning_rate:
    #     activate_decay_learning_rate()
    #
    # if img_size != 128:
    #     # TODO
    #     pass
    #     # update_image_size(new_size=img_size)

    return base_model_name, train_flag, val_flag, test_flag


def main():
    base_model_name, train_flag, val_flag, test_flag = parse_command_line_args()

    with tf.Graph().as_default():
        classifier = answerer.AnswererModel(base_model_name=base_model_name, is_trainable=True,
                                            is_primary_model=True)

        if train_flag:
            train_model(classifier)


if __name__ == '__main__':
    main()
