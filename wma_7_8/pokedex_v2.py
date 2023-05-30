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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten

#===================================================
#                       GLOBALS
#===================================================

INPUT_SHAPE = (100, 100)
OUTPUT_SIZE = 150
MAX_EPOCHS = 10

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
    train_generator = data_generator.flow_from_directory(training_dir, shuffle=True,
                                                         batch_size=batch_size, target_size=INPUT_SHAPE)
    validation_generator = data_generator.flow_from_directory(validation_dir, shuffle=True,
                                                              batch_size=batch_size, target_size=INPUT_SHAPE)
    return train_generator, validation_generator

#===================================================
#                   NEURAL NETWORKS
#===================================================

def build_example_network(input_shape, output_size):
    input_shape = (input_shape[0], input_shape[1], 3)

    input_layer = Input(shape=input_shape, name='Input')

    layer = Conv2D(256, (3, 3), padding='same', activation='relu', name='FirstConvoLargeLayer')(input_layer)
    layer = Conv2D(128, (3, 3), padding='same', activation='relu', name='FirstConvoSmallLayer')(layer)
    layer = MaxPooling2D(pool_size=(2, 2), name='FirstConvoPoolingLayer')(layer)

    layer = Conv2D(128, (3, 3), padding='same', activation='relu', name='SecondConvoLargeLayer')(layer)
    layer = Conv2D(64, (3, 3), padding='same', activation='relu', name='SecondConvoSmallLayer')(layer)
    layer = MaxPooling2D(pool_size=(2, 2), name='SecondConvoPoolingLayer')(layer)

    layer = Conv2D(64, (3, 3), padding='same', activation='relu', name='ThirdConvoLargeLayer')(layer)
    layer = Conv2D(32, (3, 3), padding='same', activation='relu', name='ThirdConvoSmallLayer')(layer)
    layer = MaxPooling2D(pool_size=(2, 2), name='ThirdConvoPoolingLayer')(layer)

    layer = Flatten()(layer)
    layer = Dense(128, activation='relu', name='HiddenDense')(layer)
    output_layer = Dense(output_size, activation='relu', name='Categories')(layer)

    model = Model(inputs=[input_layer], outputs=[output_layer], name='Example')
    return model

networks = {'example':build_example_network}

def build_network(network_name, input_size, output_size):
    try:
        network = networks[network_name](input_size, output_size)
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return network
    except KeyError:
        raise KeyError(f'No network with name {network_name}.')

#===================================================
#                 TRAINING FUNCTION
#===================================================

def train_network(network, train_generator, validation_generator, max_epochs):
    history = network.fit_generator(train_generator, epochs=max_epochs,
                                    verbose=1, validation_data=validation_generator)
    return history

#==================================================================================================
#                                       VISUALISATIONS 
#==================================================================================================

def plot_history(history, output_dir, experiment_id):
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc)) # Get number of epochs

    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.plot(epochs, acc, 'r-', label='Training')
    plt.plot(epochs, val_acc, 'b--', label='Validation')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{experiment_id}_accuracy.png'))
    plt.clf()

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(epochs, loss, 'r-', label='Training')
    plt.plot(epochs, val_loss, 'b--', label='Validation')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{experiment_id}_loss.png'))
    plt.clf()

def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Label")
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, textcolors=("black", "white")):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    threshold = im.norm(data.max()) / 2.

    # Set default alignment to center
    kw = dict(horizontalalignment="center", verticalalignment="center")

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, data[i, j], **kw)
            texts.append(text)

    return texts

def load_eng_labels(classes):
    labels = list()
    for label in classes:
        labels.append(label.split('(')[0])

    labels = np.array(labels, dtype=object)
    return labels

def print_confusion_matrix(y_pred, y_true, classifications, output_dir, experiment_id):
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(classifications))

    fig, ax = plt.subplots(figsize=(12, 10))
    im, cbar = heatmap(confusion_mtx, classifications, classifications, ax=ax,
                       cmap=plt.cm.GnBu, cbarlabel="Classification")
    annotate_heatmap(im)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{experiment_id}_confusion.png'))
    plt.clf()

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
    parser.add_argument('-n', '--network_name', required=True, type=str,
                        help='Name of the network.')
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
    train_generator, validation_generator = get_image_generators(args.train_directory, args.validation_directory, args.batch_size)
    input_shape = INPUT_SHAPE
    output_size = OUTPUT_SIZE
    network = build_network(args.network_name, input_shape, output_size)
    lprint(network.summary())
    history = train_network(network, train_generator, validation_generator, MAX_EPOCHS)
    plot_history(history, args.output_paths, args.network_name)

if __name__ == '__main__':
    main(parse_arguments())