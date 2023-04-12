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
    source_image = cv.imread(SOURCE_IMAGE_PATH)
    cv.imshow('Source', source_image)
    cv.waitKey(0)

if __name__ == '__main__':
    main()