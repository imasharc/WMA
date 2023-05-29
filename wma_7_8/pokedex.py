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
import keras as ks
## SET UP GPU WORK
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
fix_gpu()

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

    # type_encoded = pd.get_dummies(pokedex['Type1'])
    # pokedex = pd.merge(
    #     left=pokedex,
    #     right=type_encoded,
    #     left_index=True,
    #     right_index=True
    # )
    # pokedex.drop('Type1', axis=1, inplace=True)
    
    return pokedex

#===================================================
#               PREPARE DATA FOR NETWORK
#===================================================

def prepare_data_for_network(pokedex):
    data_generator = ImageDataGenerator(validation_split=0.1,
                                        rescale=1.0/255, rotation_range=30,
                                        width_shift_range=0.5, height_shift_range=0.5,
                                        zoom_range=0.5, fill_mode='nearest')
    train_generator = data_generator.flow_from_dataframe(pokedex,
                                                         x_col='Image', y_col='Type1',
                                                         subset='training', color_mode='rgba',
                                                         class_mode='categorical', target_size=(120, 120),
                                                         shuffle=True)
    test_generator = data_generator.flow_from_dataframe(pokedex,
                                                         x_col='Image', y_col='Type1',
                                                         subset='validation', color_mode='rgba',
                                                         class_mode='categorical', target_size=(120, 120),
                                                         shuffle=True)
    return train_generator, test_generator

#===================================================
#               SHOW GENERATOR RESULTS
#===================================================

def show_generator_results(generator):
    for i in range(10):
        plt.subplot(2, 5, i+1)
        for x, y in generator:
            img = x[0]
            plt.imshow(img)
            break
    plt.show()

#===================================================
#                   PREPARE NETWORK
#===================================================

def prepare_network():
    # KERAS ALLOWS TO CREATE NEURAL NETWORKS IN A SIMPLE WAY
    model = ks.models.Sequential()              # feed-forward model
    
    model.add(ks.layers.Conv2D(34, (3, 3), activation='relu', input_shape=(120, 120, 4)))
    model.add(ks.layers.MaxPooling2D(2, 2))

    model.add(ks.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(ks.layers.MaxPooling2D(2, 2))

    # FLATTENING THE MODEL
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(128, activation='relu'))
    model.add(ks.layers.Dense(18, activation='softmax'))

    # COMPILING THE MODEL
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())

    return model

#===================================================
#                   MAIN FUNCTION
#===================================================

def show_dataset_info(pokedex):
    print(pokedex.info())
    print(pokedex.head())

def main():
    args = parse_arguments()
    pokedex = load_pokedex(args.description_file, args.image_folder)
    # show_dataset_info(pokedex)
    # show_example_images(args.image_folder)
    # generator = prepare_data_for_network(pokedex)
    # show_generator_results(generator)

    train_data, test_data = prepare_data_for_network(pokedex)
    model = prepare_network()
    history = model.fit(train_data, validation_data=test_data, epochs=30)
    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:,['acc', 'val_acc']].plot()
    plt.plot(history.history['acc'])
    plt.show()

if __name__ == '__main__':
    main()