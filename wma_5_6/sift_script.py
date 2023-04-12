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
import numpy as np

#===================================================
#                       PARAMETERS
#===================================================

SOURCE_IMAGE_PATH = 'materials\lab_5_6\source.png'
TARGET_IMAGE_PATH = 'materials\lab_5_6\simple.png'
FLANN_INDEX_KDTREE = 1
FLANN_TREES = 5
FLANN_CHECKS = 50
FLANN_K = 2
KEYPOINT_VALIDITY_THRESHOLD = 0.7

#===================================================
#                   MAIN FUNCTION
#===================================================

def main():

    sift = cv.SIFT_create()

    source_image = cv.imread(SOURCE_IMAGE_PATH)
    gray_source = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)
    
    source_keypoints, source_descriptors = sift.detectAndCompute(gray_source, None)
    marked_source = cv.drawKeypoints(source_image, source_keypoints, None)
    cv.imshow('Source', marked_source)

    target_image = cv.imread(TARGET_IMAGE_PATH)
    gray_target = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)
    
    target_keypoints, target_descriptors = sift.detectAndCompute(gray_target, None)
    marked_target = cv.drawKeypoints(target_image, target_keypoints, None)
    cv.imshow('Target', marked_target)

    index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': FLANN_TREES}
    search_params = {'checks': FLANN_CHECKS}
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(source_descriptors, target_descriptors, k = FLANN_K)

    # matches_mask = [[0, 0] for i in range(len(matches))]
    matches_mask = []
    valid_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < KEYPOINT_VALIDITY_THRESHOLD * n.distance:
            # matches_mask[i] = [1, 0]
            matches_mask.append([m])
            # valid_matches.append(target_keypoints[i])
    # valid_matches = np.asarray(valid_matches, dtype = np.int32)

    draw_parameters = {'matchColor': (0, 0, 255), 'singlePointColor': (255, 0, 0),
                       'matchesMask': matches_mask, 'flags': cv.DrawMatchesFlags_DEFAULT}
    matches_visualisation = cv.drawMatchesKnn(source_image, source_keypoints,
                                              target_image, target_keypoints,
                                              matches_mask, None,
                                              flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('Matches', matches_visualisation)
    
    print(source_descriptors[0])
    cv.waitKey(0)

if __name__ == '__main__':
    main()