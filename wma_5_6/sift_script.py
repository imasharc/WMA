#===================================================
#                     DESCRIPTION
#===================================================

# The program aims to implement SIFT algorithm from scratch,
# which is a feature detection algorithm in Computer Vision
# (Scale Invariant Feature Transform)
# above info source: https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/

# The script takes the source file and matches its keypoints
# onto the target file to detect the object from the target file.

#===================================================
#                       IMPORTS
#===================================================

import cv2 as cv
import numpy as np
from sklearn.cluster import AffinityPropagation

#===================================================
#                       PARAMETERS
#===================================================

# SOURCE_IMAGE_PATH = 'materials\lab_5_6\source.png'
# TARGET_IMAGE_PATH = 'materials\lab_5_6\challenge.png'
FLANN_INDEX_KDTREE = 1
FLANN_TREES = 5
FLANN_CHECKS = 50
FLANN_K = 2
KEYPOINT_VALIDITY_THRESHOLD = 0.675
AFFIINITY_DAMPING = 0.9

#===================================================
#                 SIFT ALGORITHM
#===================================================

def sift(source_image_path, target_image_path):
    sift = cv.SIFT_create()

    source_image = cv.imread(source_image_path)
    gray_source = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)
    
    source_keypoints, source_descriptors = sift.detectAndCompute(gray_source, None)
    marked_source = cv.drawKeypoints(source_image, source_keypoints, None)
    # cv.imshow('Source', marked_source)

    target_image = cv.imread(target_image_path)
    gray_target = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)
    
    target_keypoints, target_descriptors = sift.detectAndCompute(gray_target, None)
    marked_target = cv.drawKeypoints(target_image, target_keypoints, None)
    # cv.imshow('Target', marked_target)

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
            valid_matches.append(target_keypoints[matches[i][0].trainIdx].pt)
    valid_matches = np.asarray(valid_matches, dtype = np.int32)

    draw_parameters = {'matchColor': (0, 0, 255), 'singlePointColor': (255, 0, 0),
                       'matchesMask': matches_mask, 'flags': cv.DrawMatchesFlags_DEFAULT}
    matches_visualisation = cv.drawMatchesKnn(source_image, source_keypoints,
                                              target_image, target_keypoints,
                                              matches_mask, None,
                                              flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv.imshow('Matches', matches_visualisation)

    detected_visualisation = target_image.copy()
    for point in valid_matches:
        cv.circle(detected_visualisation, tuple(point), 5, [0, 0, 255])
    
    af = AffinityPropagation(damping = AFFIINITY_DAMPING).fit(valid_matches)
    cluster_center_indices = af.cluster_centers_indices_
    labels = af.labels_
    cluster_count = len(cluster_center_indices)
    for cluster in range(cluster_count):
        cluster_points = valid_matches[labels == cluster]
        x_min, y_min = np.min(cluster_points, axis = 0)
        x_max, y_max = np.max(cluster_points, axis = 0)
        cv.rectangle(detected_visualisation, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    cv.imshow('Detected', detected_visualisation)
    
    print(source_descriptors[0])
    cv.waitKey(0)

#===================================================
#                   MAIN FUNCTION
#===================================================

def main():
    sift('materials\lab_5_6\source.png', 'materials\lab_5_6\challenge.png')

if __name__ == '__main__':
    # sift(SOURCE_IMAGE_PATH, TARGET_IMAGE_PATH)
    main()