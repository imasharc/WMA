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
#                   MAIN FUNCTION
#===================================================

def main():
    print('Hello')

if __name__ == "__main__":
    main()