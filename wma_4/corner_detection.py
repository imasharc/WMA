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
    ax.imshow(img, cmap = 'Greys_r')
    plt.show()

#===================================================
#                   MAIN FUNCTION
#===================================================

def main():
    video = cv2.VideoCapture("materials/spongebob.mp4")

    while (video.isOpened()):
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
        ])
        kernel_y = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ])
        derrivative_x = sig.convolve2d(gray, kernel_x, mode='same')
        derrivative_y = sig.convolve2d(gray, kernel_y, mode='same')
        Ixx = ndi.gaussian_filter(derrivative_x ** 2, sigma = 1)
        Ixy = ndi.gaussian_filter(derrivative_y * derrivative_x, sigma = 1)
        Iyy = ndi.gaussian_filter(derrivative_y ** 2, sigma = 1)
        # parameter for Harris corner detection
        k = 0.05
        detA = Ixx * Iyy - Ixy ** 2
        traceA = Ixx + Iyy
        harris_response = detA - k * traceA ** 2
        # print(f'Min = {harris_response.min()} Max = {harris_response.max()}')
        harris_response_range = harris_response.max() - harris_response.min()
        scaled_response = (harris_response / harris_response_range) * 255

        # show_img(scaled_response)
        corners = np.copy(frame)
        edges = np.copy(frame)
        h_min = harris_response.min()
        h_max = harris_response.max()
        THRESHOLD_CORNER = 0.01
        THRESHOLD_EDGE = 0.01

        if ret == True:
            for y, row in enumerate(harris_response):
                for x, pixel in enumerate(row):
                    if pixel > h_max * THRESHOLD_CORNER:
                        corners[y, x] = [0, 0, 255]
                    elif pixel <= h_min * THRESHOLD_EDGE:
                        edges[y, x] = [255, 255, 255]
        
        res = cv2.addWeighted(corners, 0.5, edges, 0.5, 0)
        cv2.imshow('spongebob', res)
        # cv2.imshow('spongebob', edges)
        # cv2.imshow('EDGES', canny_output)

        if cv2.waitKey(25) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    # show_img(corners)
    # show_img(edges)

if __name__ == '__main__':
    main()