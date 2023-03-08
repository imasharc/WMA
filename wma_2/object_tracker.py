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

            canny_output = cv2.Canny(red, 100, 200)
            red_contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            contours_poly = [None]*len(red_contours)
            boundRect = [None]*len(red_contours)
            for i, c in enumerate(red_contours):
                contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                boundRect[i] = cv2.boundingRect(contours_poly[i])

            for i in range(len(red_contours)):
                color = (rng.randint(0, 256), rng.randint(
                    0, 256), rng.randint(0, 256))
                # cv2.drawContours(frame, contours_poly, i, color)
                if abs(boundRect[i][0]-boundRect[i][1]) < 200 and abs(boundRect[i][2]-boundRect[i][3]) < 200:
                    cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])),
                                (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)

            cv2.imshow('Frame', frame)
            cv2.imshow('red', red)
            # cv2.imshow('MASK', black_stuff)
            cv2.imshow('EDGES', canny_output)

            if cv2.waitKey(25) == ord('q'):
                break

        else:
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    display_video()