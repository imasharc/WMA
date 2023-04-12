#===================================================
#                     DESCRIPTION
#===================================================

# The program aims to implement SIFT algorithm from scratch,
# which is a feature detection algorithm in Computer Vision
# (Scale Invariant Feature Transform)
# above info source: https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/

#===================================================
#                       IMPORTS
#===================================================

import cv2 as cv

#===================================================
#                   CONFIGURATION
#===================================================

SOURCE_IMAGE_PATH = 'materials\lab_5_6\source.png'

#===================================================
#                   MAIN FUNCTION
#===================================================

def main():

    sift = cv.SIFT_create()

    source_image = cv.imread(SOURCE_IMAGE_PATH)
    gray_source = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)
    cv.imshow('Source', source_image)
    
    source_keypoints, source_descriptors = sift.detectAndCompute(gray_source, None)
    marked_source = cv.drawKeypoints(source_image, source_keypoints, None)
    cv.imshow('Key points', marked_source)
    print(source_descriptors[0])
    cv.waitKey(0)

if __name__ == '__main__':
    main()