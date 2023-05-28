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
import argparse

#===================================================
#                   MAIN FUNCTION
#===================================================

def show_dataset_info(pokedex):
    print(pokedex.info())
    print(pokedex.head())

def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--description_file', default='pokemon/pokemon.csv', help='A CSV file with pokemon informations')
    parser.add_argument('-i', '--image_path', default='pokemon/images/images', help='Folder with pokemon images')
    return parser.parse_args()

def main():
    args = parse_arguments()
    pokedex = pd.read_csv(args.description_file)
    show_dataset_info(pokedex)

if __name__ == '__main__':
    main()