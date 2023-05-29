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

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import argparse
import os               # provides operating system specific functions
from keras.preprocessing.image import ImageDataGenerator

#===================================================
#                   ARGUMENT PARSER
#===================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--description_file', default='pokemon\pokemon.csv', help='A CSV file with pokemon informations')
    parser.add_argument('-i', '--image_folder', default='pokemon\images', help='Folder with pokemon images')
    return parser.parse_args()

#===================================================
#               DISPLAY POKEMON IMAGES
#===================================================

def show_example_images(image_folder):
    images = sorted(os.listdir(image_folder))
    fig, axis = plt.subplots(2, 4)
    axis = axis.flatten()
    for idx, img_file in enumerate(images):
        if idx >= len(axis):
            break
        img = mimg.imread(os.path.join(image_folder, img_file))
        axis[idx].imshow(img)
        axis[idx].set_title(img_file.split('.')[0])
        axis[idx].axis('off')
    plt.show()

#===================================================
#                   LOAD POKEDEX
#===================================================

def load_pokedex(description_file, image_folder):
    pokedex = pd.read_csv(description_file)
    pokedex.drop('Type2', axis=1, inplace=True)
    pokedex.sort_values(by=['Name'], ascending=True, inplace=True)

    images = sorted(os.listdir(image_folder))
    images = list(map(lambda image_file: os.path.join(image_folder, image_file), images))
    pokedex['Image'] = images

    type_encoded = pd.get_dummies(pokedex['Type1'])
    pokedex = pd.merge(
        left=pokedex,
        right=type_encoded,
        left_index=True,
        right_index=True
    )
    pokedex.drop('Type1', axis=1, inplace=True)
    
    return pokedex

#===================================================
#
#===================================================

def prepare_data_for_network(pokedex):
    data_generator = ImageDataGenerator(validation_split=0.1)
    return data_generator

#===================================================
#                   MAIN FUNCTION
#===================================================

def show_dataset_info(pokedex):
    print(pokedex.info())
    print(pokedex.head())

def main():
    args = parse_arguments()
    pokedex = load_pokedex(args.description_file, args.image_folder)
    show_dataset_info(pokedex)
    show_example_images(args.image_folder)
    prepare_data_for_network(pokedex)

if __name__ == '__main__':
    main()