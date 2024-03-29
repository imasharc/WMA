#===================================================
#                     DESCRIPTION
#===================================================

# The program aims to track the object from the target file
# by using the SIFT algorithm and the Affinity Propagation.

# The first step is the 'training stage' where the source images are matched
# and the valid keypoints are taken from them. Initially two images are matched,
# after the initial matching only the first-best keypoints and the next image are matched together.
# After the 'training stage', the 'testing stage' is initialized by matching 'the best' keypoints
# from the 'training stage' with the keypoints extracted from the individual video frames.
# Finally, matching keypoints are marked on the video frames themselves resulting
# in detecting the object on the target video

#===================================================
#                       IMPORTS
#===================================================

import cv2
import numpy as np
from sklearn.cluster import AffinityPropagation

#===================================================
#                     PARAMETERS
#===================================================

SOURCE_IMAGE_PATH = 'materials\lab_5_6\saw1.jpg'
SOURCE_IMAGE_PATH_2 = 'materials\lab_5_6\saw2.jpg'
SOURCE_IMAGE_PATH_3 = 'materials\lab_5_6\saw3.jpg'
SOURCE_IMAGE_PATH_4 = 'materials\lab_5_6\saw4.jpg'
TARGET_IMAGE_PATH = 'materials\lab_5_6\saw4.jpg'
TARGET_VIDEO_PATH = 'materials\lab_5_6\sawmovie.mp4'
FLANN_INDEX_KDTREE = 1
FLANN_TREES = 5
FLANN_CHECKS = 50
FLANN_K = 2
KEYPOINT_VALIDITY_THRESHOLD = 0.68
AFFIINITY_DAMPING = 0.94

#===================================================
#                   TRAINING STAGE
#===================================================

def sift():
    sift = cv2.SIFT_create()

    source_image = cv2.imread(SOURCE_IMAGE_PATH)
    source_image = cv2.resize(source_image, dsize=(600, 600))
    gray_source = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

    source_keypoints, source_descriptors = sift.detectAndCompute(gray_source, None)
    marked_target = cv2.drawKeypoints(source_image, source_keypoints, None)
    cv2.imshow('Target', marked_target)
    
    source_image_2 = cv2.imread(SOURCE_IMAGE_PATH_2)
    source_image_2 = cv2.resize(source_image_2, dsize=(600, 600))
    gray_source_2 = cv2.cvtColor(source_image_2, cv2.COLOR_BGR2GRAY)

    source_keypoints_2, source_descriptors_2 = sift.detectAndCompute(gray_source_2, None)
    marked_target_2 = cv2.drawKeypoints(source_image_2, source_keypoints_2, None)
    cv2.imshow('Target_2', marked_target_2)

    index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': FLANN_TREES}
    search_params = {'checks': FLANN_CHECKS}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(source_descriptors, source_descriptors_2, k=FLANN_K)

    print(matches)

    matches_mask = []
    valid_matches = []
    for i, (m,n) in enumerate(matches):
        if m.distance < KEYPOINT_VALIDITY_THRESHOLD*n.distance:
            matches_mask.append([m])
            valid_matches.append(source_keypoints_2[matches[i][0].trainIdx].pt)
    valid_matches = np.asarray(valid_matches, dtype=np.int32)

    draw_parameters = {'matchColor': (0,0,255), 'singlePointColor': (255,0,0),
                       'matchesMask': matches_mask, 'flags':cv2.DrawMatchesFlags_DEFAULT}
    matches_visualisation = cv2.drawMatchesKnn(source_image, source_keypoints,
                                              source_image_2, source_keypoints_2, 
                                              matches_mask, None, 
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', matches_visualisation)
    
    cv2.waitKey()

#===================================================
#                   MAIN FUNCTION
#===================================================

def main():
    sift()

if __name__ == "__main__":
    main()