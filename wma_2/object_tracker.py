'''
This is a script that will track red object on the screen.
'''
import cv2
import numpy as np

def display_video():
    cap = cv2.VideoCapture("materials/rgb_balls.mp4")
    
    if (cap.isOpened() == False):
        print("ERROR")
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)

            if cv2.waitKey(25) == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    display_video()