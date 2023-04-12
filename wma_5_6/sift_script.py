import cv2 as cv
SOURCE_IMAGE_PATH = 'materials\lab_5_6\source.png'

def main():
    source_image = cv.imread(SOURCE_IMAGE_PATH)
    cv.imshow('Source', source_image)
    cv.waitKey(0)

if __name__ == '__main__':
    main()