#===================================================
#                     DESCRIPTION
#===================================================

# The program aims to implement a Harris Corner Detection algorithm
# from scratch using numpy library.
# It should take a video as an input and show
# only edges as white lines and corners as red dots.

#===================================================
#                       IMPORTS
#===================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy import ndimage as ndi

#===================================================
#                   IMAGE DISPLAY
#===================================================

def show_img(img, bw = False):
    fig = plt.figure(figsize = (13, 13))
    ax = fig.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(img, cmap = 'Greys_r' if bw else None)
    plt.show()

#===================================================
#                   MAIN FUNCTION
#===================================================

def main():
    img = cv2.imread('materials/box_2.png')
    show_img(img)

if __name__ == '__main__':
    main()