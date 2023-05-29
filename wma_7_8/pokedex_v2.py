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

#===================================================
#                       LOGGING
#===================================================

logger = logging.getLogger()
lprint = logger.info
wprint = logger.warning

def init_logger(output_dir):
    log_formatter = logging.Formatter('%(message)')
    logfile_path = os.path.join(output_dir, 'train_pokedex_v2_convo_net.log')
    file_handler = logging.FileHandler(logfile_path)
    
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    
    consoler_handler = logging.StreamHandler()
    consoler_handler.setFormatter(log_formatter)
    logger.addHandler(consoler_handler)

    logger.setLevel(logging.INFO)

#===================================================
#                   ARGUMENT PARSER
#===================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--description_file', default='pokemon\pokemon.csv', help='A CSV file with pokemon informations')
    parser.add_argument('-i', '--image_folder', default='pokemon\images', help='Folder with pokemon images')
    return parser.parse_args()

#===================================================
#                     MAIN FUNCTION
#===================================================

def main(args):
    pass

if __name__ == '__main__':
    main(parse_arguments())