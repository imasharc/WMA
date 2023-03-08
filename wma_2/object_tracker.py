'''
This is a script that will track red object on the screen.
'''
import cv2
import numpy as np
import random as rng
import math

def display_video():
    rng.seed(12345)
    CUTOFF = 60
    video = cv2.VideoCapture("materials/rgb_balls.mp4")
    
    if (video.isOpened() == False):
        print("ERROR")
    
    while (video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            hsv_red = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # define red color range
            low_red = np.array([0, 200, 100])
            high_red = np.array([10, 255, 255])
            red_mask = cv2.inRange(hsv_red, low_red, high_red)
            red = cv2.bitwise_and(frame, frame, mask=red_mask)

            cv2.imshow('Frame', frame)
            cv2.imshow('red', red)
            # cv2.imshow('MASK', black_stuff)
            # cv2.imshow('EDGES', canny_output)

            if cv2.waitKey(25) == ord('q'):
                break

        else:
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    display_video()