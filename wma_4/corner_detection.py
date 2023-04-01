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
    # video = cv2.VideoCapture("materials/rgb_ball_720.mp4")
    video = cv2.VideoCapture("spongebob.mp4")

    if (video.isOpened() == False):
        print("SOMETHING WENT WRONG...THE PROGRAM WILL RELOAD THE VIDEO PATH")
        video = cv2.VideoCapture("materials/spongebob.mp4")
        print("ALL DONE")
        print("YOUR RELATIVE PATH IS AS FOLLOWS:\nmaterials/spongebob.mp4")
        print("\nPRESS 'q' TO CLOSE THE WINDOW")
    if (video.isOpened() == False):
        print("SOMETHING WENT WRONG...THE PROGRAM NEEDS YOU TO PROVIDE A RELATIVE PATH TO YOUR VIDEO\n")
        path_input = input("Enter relative path to your video:")
        video = cv2.VideoCapture(path_input)
        print("ALL DONE")
        print("YOUR RELATIVE PATH IS AS FOLLOWS:\n", path_input)
        print("\nPRESS 'q' TO CLOSE THE WINDOW")

    while (video.isOpened()):
        ret, frame = video.read()

        if ret == True:
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
            THRESHOLD_CORNER = 0.1
            THRESHOLD_EDGE = 0.0001

            for y, row in enumerate(harris_response):
                for x, pixel in enumerate(row):
                    if pixel > h_max * THRESHOLD_CORNER:
                        frame[y, x] = [0, 0, 255]
                    elif pixel <= h_min * THRESHOLD_EDGE:
                        frame[y, x] = [255, 255, 255]
        
        # res = cv2.addWeighted(corners, 0.5, edges, 0.5, 0)
        cv2.imshow('harris_corner_edge_detection', frame)
        # cv2.imshow('spongebob', edgeqs)
        # cv2.imshow('EDGES', canny_output)

        if cv2.waitKey(25) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    # show_img(corners)
    # show_img(edges)

if __name__ == '__main__':
    main()