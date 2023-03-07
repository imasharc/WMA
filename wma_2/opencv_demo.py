import cv2
import numpy as np
import random as rng
import math

def main():
    pass

def display_image():
    image = cv2.imread('materials\example.jpg')
    print(image.shape)
    part = image[500:800, 400:1000, 0]
    mask = part > 200
    part[mask] = 0
    cv2.imshow('Our window', image)
    cv2.waitKey(0)


def main2():
    rng.seed(12345)
    CUTOFF = 60
    video = cv2.VideoCapture('\materials\example.mp4')
    while (video.isOpened()):
        ret, frame = video.read()
        bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_stuff = np.zeros_like(bw)
        black_stuff[bw < CUTOFF] = 255

        kernel = np.ones((5, 5), np.uint8)
        black_stuff = cv2.erode(black_stuff, kernel, iterations=2)
        black_stuff = cv2.dilate(black_stuff, kernel, iterations=5)

        canny_output = cv2.Canny(black_stuff, 100, 200)
        contours, _ = cv2.findContours(
            canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])

        for i in range(len(contours)):
            color = (rng.randint(0, 256), rng.randint(
                0, 256), rng.randint(0, 256))
            # cv2.drawContours(frame, contours_poly, i, color)
            if abs(boundRect[i][0]-boundRect[i][1]) < 200 and abs(boundRect[i][2]-boundRect[i][3]) < 200:
                cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])),
                              (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)

        # frame[bw<CUTOFF,2] = 255
        cv2.imshow('Our video', frame)
        cv2.imshow('MASK', black_stuff)
        cv2.imshow('EDGES', canny_output)
        if cv2.waitKey(10) == ord('q'):
            break

def display_video():
    # VideoCapture(0) for camera
    CUTOFF = 80
    video = cv2.VideoCapture('materials/rgb_balls.mp4') 
    while(video.isOpened()):
        ret, frame = video.read()
        bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_stuff = np.zeros_like(bw)
        frame[bw<CUTOFF,2] = 255
        cv2.imshow('Our video', frame)
        if cv2.waitKey(10) == ord('q'):
            break

if __name__ == '__main__':
    display_video()
    # main2()
