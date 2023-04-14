#===================================================
#                     DESCRIPTION
#===================================================

# The program aims to track the object from the target file
# by using the SIFT algorithm implementation from the script
# The program takes multiple images as source files
# and matches their keypoints onto the target file (video file)

#===================================================
#                       IMPORTS
#===================================================

import cv2 as cv
import numpy as np
from sklearn.cluster import AffinityPropagation

#===================================================
#                       PARAMETERS
#===================================================

SOURCE_IMAGE_PATH = 'materials\lab_5_6\saw1.jpg'
SOURCE_IMAGE_PATH_2 = 'materials\lab_5_6\saw2.jpg'
TARGET_IMAGE_PATH = 'materials\lab_5_6\saw3.jpg'
TARGET_VIDEO_PATH = 'materials\lab_5_6\sawmovie.mp4'
FLANN_INDEX_KDTREE = 1
FLANN_TREES = 5
FLANN_CHECKS = 50
FLANN_K = 2
KEYPOINT_VALIDITY_THRESHOLD = 0.85
AFFIINITY_DAMPING = 0.9

#===================================================
#                   SIFT ALGORITHM
#===================================================

def show_video(video_path):
    target_video = cv.VideoCapture(video_path)

    cap = target_video
 
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        
            sift(SOURCE_IMAGE_PATH, frame)
            cv.imshow('Frame',frame)
        
            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            break
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv.destroyAllWindows()

def sift(source_image_path, source_image_path_2, target_image_path):
    sift = cv.SIFT_create()

    source_image = cv.imread(source_image_path)
    source_image = cv.resize(source_image, dsize=(600, 600))
    gray_source = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)

    source_image_2 = cv.imread(source_image_path_2)
    source_image_2 = cv.resize(source_image_2, dsize=(600, 600))
    gray_source_2 = cv.cvtColor(source_image_2, cv.COLOR_BGR2GRAY)
    
    source_keypoints, source_descriptors = sift.detectAndCompute(gray_source, None)
    marked_source = cv.drawKeypoints(source_image, source_keypoints, None)
    cv.imshow('Source_1', marked_source)

    source_keypoints_2, source_descriptors_2 = sift.detectAndCompute(gray_source_2, None)
    marked_source_2 = cv.drawKeypoints(source_image_2, source_keypoints_2, None)
    cv.imshow('Source_2', marked_source_2)

    sources_keypoints = np.concatenate((source_keypoints, source_keypoints_2))
    sources_descriptors = np.concatenate((source_descriptors, source_descriptors_2))
    marked_sources = cv.drawKeypoints(source_image_2, sources_keypoints, None)
    print(sources_descriptors)
    cv.imshow('Sources', marked_sources)

    target_image = cv.imread(target_image_path)
    target_image = cv.resize(target_image, dsize=(600, 600))
    gray_target = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)
    
    target_keypoints, target_descriptors = sift.detectAndCompute(gray_target, None)
    marked_target = cv.drawKeypoints(target_image, target_keypoints, None)
    # cv.imshow('Target', marked_target)

    index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': FLANN_TREES}
    search_params = {'checks': FLANN_CHECKS}
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(source_descriptors, target_descriptors, k = FLANN_K)
    # print(source_descriptors)
    # print(matches)

    # matches_mask = [[0, 0] for i in range(len(matches))]
    matches_mask = []
    valid_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < KEYPOINT_VALIDITY_THRESHOLD * n.distance:
            # matches_mask[i] = [1, 0]
            matches_mask.append([m])
            valid_matches.append(target_keypoints[matches[i][0].trainIdx].pt)
    valid_matches = np.asarray(valid_matches, dtype = np.int32)
    # print(valid_matches)

    draw_parameters = {'matchColor': (0, 0, 255), 'singlePointColor': (255, 0, 0),
                       'matchesMask': matches_mask, 'flags': cv.DrawMatchesFlags_DEFAULT}
    matches_visualisation = cv.drawMatchesKnn(source_image, source_keypoints,
                                              target_image, target_keypoints,
                                              matches_mask, None,
                                              flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matches_visualisation = cv.resize(matches_visualisation, dsize=(600, 600))
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
    
    # print(source_descriptors[0])
    cv.waitKey(0)

#===================================================
#                   MAIN FUNCTION
#===================================================

def main():
    sift(SOURCE_IMAGE_PATH, SOURCE_IMAGE_PATH_2, TARGET_IMAGE_PATH)
    # show_video(TARGET_VIDEO_PATH)

if __name__ == '__main__':
    # sift(SOURCE_IMAGE_PATH, TARGET_IMAGE_PATH)
    main()