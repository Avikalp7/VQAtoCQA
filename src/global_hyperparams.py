"""Module containing hyperparameters and training settings
"""
# Copyright (C) 2018 Yahoo Japan Corporation (Licensed under CC BY-NC-SA 4.0)
# https://creativecommons.org/licenses/by-nc-sa/4.0/


from enum import Enum

# Whether to use reduced vocabulary data, where low frequency words have been removed.
USE_TRUNC_VOCAB = True
# Whether the task is multi-label classification or single-label classification
USE_MULTILABEL = True
# Whether to use data where some confusing labels such as 'Computer Tech' and 'Internet, PC, Appliances' are merged.
USE_MERGED_LABELS = True
# Use data from 2014, not both 2013 and 2014
USE_2014_DATA = True
# Let the backpropagation updates be applied to last resnet block or not.
TRAIN_RESNET_LAST_BLOCK = False

# Number of classes for classification
NUM_CLASSES = 38 if USE_2014_DATA else 52

# optimizer to use in gradient descent
OptimizerType = Enum('Optimizer', 'adam rms_optimizer')
OPTIMIZER = OptimizerType.adam

# TRAINING PARAMETERS #
RESNET_LAYERS = 50
decay_learning_rate = False
scale_learning_rate = False
early_stopping_learning_rate = True
use_batch_norm = True

# Whether to use gradient clipping
USE_GRADIENT_CLIPPING = True

# IMAGE HYPERPARAMS #
IMAGE_SIZE = 224
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3
# ----------------- #

# TEXT HYPERPARAMS #
TEXT_LENTH = 160
WORD_SIZES = (305348 + 2) if not USE_TRUNC_VOCAB else 48900
EMBED_SIZES = 256 if not USE_2014_DATA else 100

# Enums for the type and names of model we have
ModelType = Enum('ModelType', 'text_only image_only image_text answerer')

ModelName = Enum('ModelName', 'text_cnn bi_lstm text_aug temp_text text_rnn hie_text '
                              'image_cnn image_cnn_v2 resnet resnet_clf '
                              'stacked_attention_cnn stacked_attention_with_freeze_cnn hie_co_att text_to_image_match '
                              'stacked_attention_with_semi_freeze_cnn embedding_concat my_model '
                              'adaptive_attention_with_freeze_cnn MCB embedding_concat_semifreeze aux_task_model'
                              'answerer')

# ANSWERER MODEL PARAMETERS #
ANSWERER_EMBED_DIM = 128
NUM_ANSWERERS = 1275

learning_rate_dict = {
    ModelType.text_only: 2e-3,
    ModelType.image_only: 2e-3,
    ModelType.image_text: 2e-3,
    ModelType.answerer: 2e-3
}

DECAY_STEP_LEARNING_RATE = 3000
DECAY_RATE_LEARNING_RATE = 0.90

scale_factors_dict = {
    ModelType.image_text: [1, 0.6, 0.3, 0.1, 0.05, 0.02, 5e-3, 1e-4, 5e-5],
    ModelType.image_only: [1, 0.5, 0.2, 0.1, 2e-2, 2e-3, 2e-4],
    ModelType.text_only: [1, 0.5, 0.1, 0.02, 0.002, 2e-4, 2e-5]
}

scale_epochs_dict = {
    ModelType.image_text: [1, 3, 6, 9, 12, 15, 18, 20],
    ModelType.image_only: [2, 8, 12, 16, 19, 22],
    ModelType.text_only: [2, 5, 8, 12, 15, 18]
}

batch_size_dict = {
    ModelType.text_only: 32*2,
    ModelType.image_only: 32*2,
    ModelType.image_text: 32*2,
    ModelType.answerer: 32*2
}

training_epochs_dict = {
    ModelType.text_only: 300,
    ModelType.image_only: 300,
    ModelType.image_text: 300,
    ModelType.answerer: 300,
}

best_date = {
    ModelName.text_cnn: '18Y07M18D10h15m',
    ModelName.image_cnn_v2: '18Y06M26D08h08m',
    ModelName.my_model: '18Y06M22D06h51m',
    ModelName.stacked_attention_with_semi_freeze_cnn: '18Y07M30D05h10m',
    ModelName.embedding_concat_semifreeze: '18Y07M20D08h25m',
}

best_epochs = {
    ModelName.text_cnn: 8,
    ModelName.image_cnn_v2: 11,
    ModelName.my_model: 12,
    ModelName.stacked_attention_with_semi_freeze_cnn: 25,
    ModelName.embedding_concat_semifreeze: 18,
}

NUM_TEXT_IN_MULTI_CHOICE = 5
NUM_SAME_CAT_IN_MULTI_CHOICE = 1

WEIGHT_DECAY = 0.00004 if not USE_2014_DATA else 0.0001
OPTIMIZER_MOMENTUM = 0.99

NUM_IMAGES = {
    'train': 815066,
    'validation': 101884,
    'test': 101883
}


global_config = {
    'USE_2014_DATA': USE_2014_DATA,
    'USE_MULTILABEL': USE_MULTILABEL,
    'USE_TRUNC_VOCAB': USE_TRUNC_VOCAB,
    'USE_MERGED_LABELS': USE_MERGED_LABELS,
    'USE_GRADIENT_CLIPPING': USE_GRADIENT_CLIPPING,
    'NUM_CLASSES': NUM_CLASSES,
    'use_batch_norm': use_batch_norm,
    'scale_learning_rate': scale_learning_rate,
    'decay_learning_rate': decay_learning_rate,
    'WEIGHT_DECAY': WEIGHT_DECAY,
}


def update_learning_rate_dict(model_type, learning_rate):
    learning_rate_dict[model_type] = learning_rate


def update_batch_size_dict(model_type, batch_size):
    batch_size_dict[model_type] = batch_size


def update_training_epochs_dict(model_type, training_epochs):
    training_epochs_dict[model_type] = training_epochs
