""" Module providing paths for data. Please change this file to be able to load your own dataset.
"""

from datetime import datetime
from global_hyperparams import ModelType

data_directory = '../experimental_data/'
results_directory = '../results/'
images_directory = '../../vqa_hliu/liu_exist_images/'
image_match_data_subdirectory = 'image_match_data/'
bottleneck_data_directory = data_directory + 'bottleneck_data/'

image_results_subdirectory = results_directory + ModelType.image_only.name + "/"
text_results_subdirectory = results_directory + ModelType.text_only.name + "/"
image_text_results_subdirectory = results_directory + ModelType.image_text.name + "/"
answerer_results_subdirectory = results_directory + ModelType.answerer.name + "/"

resnet_model_prefix = "./resnet/tensorflow-resnet-pretrained-20160509/"

# Get the current timestamp, which will be the folder name storing all weights
TIME_STAMP = datetime.now().strftime('%yY%mM%dD%Hh%Mm')
print('Timestamp: ', TIME_STAMP)

# These suffix should be appended to (image_results_subdirectory + model_name)
train_log_directory_suffix = '/train/' + TIME_STAMP + '/'
checkpoint_directory_suffix = '/ckpt/' + TIME_STAMP + '/'
test_log_directory_suffix = '/test/' + TIME_STAMP + '/'


def get_stored_checkpoint_filename(model_type, model_name, date, num_epochs):
    return results_directory + model_type.name + '/' + model_name.name + '/ckpt/' + date + '/' + \
           str(num_epochs) + 'epoch/model.ckpt'


def get_stored_metagraph_filename(model_type, model_name, date, num_epochs):
    return results_directory + model_type.name + '/' + model_name.name + '/ckpt/' + date + '/' + \
           str(num_epochs) + 'epoch/model.ckpt.meta'


def get_bottleneck_data_subdirectory(model_name):
    return bottleneck_data_directory + model_name.name + '/'


def get_resnet_stored_filename(file_type, num_layers=152):
    if file_type != 'meta' and file_type != 'ckpt':
        raise ValueError("Invalid file_type arg = %s, only 'meta' and 'ckpt' allowed" % file_type)
    return resnet_model_prefix + 'ResNet-L%d.%s' % (num_layers, file_type)
