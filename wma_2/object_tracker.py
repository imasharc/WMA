'''
This is a script that will track red object on the screen.
'''
import cv2

def display_image():
    image = cv2.imread('materials\example.jpg')
    window_name = 'image'
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    display_image()