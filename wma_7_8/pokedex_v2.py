#===================================================
#                     DESCRIPTION
#===================================================

'''
This is a script that trains Convolutional Neural Networks for image recognition.
It was created with pokemon dataset in mind but can be used for other datasets.
'''

#===================================================
#                       IMPORTS
#===================================================

import argparse
import logging
import os
from keras.preprocessing.image import ImageDataGenerator

#===================================================
#                       LOGGING
#===================================================

logger = logging.getLogger()
lprint = logger.info
wprint = logger.warning
eprint = logger.error

def init_logger(output_dir):
    log_formatter = logging.Formatter('%(message)s')
    logfile_path = os.path.join(output_dir, 'train_pokedex_v2_convo_net.log')
    file_handler = logging.FileHandler(logfile_path)
    
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    
    consoler_handler = logging.StreamHandler()
    consoler_handler.setFormatter(log_formatter)
    logger.addHandler(consoler_handler)

    logger.setLevel(logging.INFO)

#===================================================
#                 DATA HANDLING
#===================================================
    
    # batch size defines the number of images to be put at once into the model to train it
def get_image_generators(training_dir, validation_dir, batch_size):
    # template of what steps have to be done when generating the data
    # right now, ImageDataGenerator is a default one
    data_generator = ImageDataGenerator()
    # here, we are creating a specific generator for specific folders
    train_generator = data_generator.flow_from_directory(training_dir, shuffle=True, batch_size=batch_size)
    validation_generator = data_generator.flow_from_directory(validation_dir, shuffle=True, batch_size=batch_size)
    return train_generator, validation_generator

#===================================================
#                   NEURAL NETWORKS
#===================================================

def build_example_network():
    pass

networks = {'example':build_example_network}

def build_network(network_name):
    try:
        network = networks[network_name]
    except KeyError:
        raise KeyError(f'No network with name {network_name}.')

#===================================================
#                   ARGUMENT PARSER
#===================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output_path', required=True, type=str,
                        help='Directory to which all output will be saved.')
    parser.add_argument('-t', '--train_directory', required=True, type=str,
                        help='Directory with training data')
    parser.add_argument('-v', '--validation_directory', required=True, type=str,
                        help='Directory with validation data')
    parser.add_argument('-b', '--batch_size', default=20, type=positive_integer,
                        help='Size of a single training batch')
    return parser.parse_args()

#===================================================
#               BATCH SIZE CONSTRAINTS
#===================================================

def positive_integer(txt):
    try:
        value = int(txt)
        if value <= 0:
            raise ValueError(f'Value {value} is not a positive integer.')
    except ValueError as err:
        raise ValueError(f'Unable to convert {txt} to integer.')

#===================================================
#                     MAIN FUNCTION
#===================================================

def main(args):
    init_logger(args.output_path)
    get_image_generators(args.train_directory, args.validation_directory, args.batch_size)
    print('It works')

if __name__ == '__main__':
    main(parse_arguments())