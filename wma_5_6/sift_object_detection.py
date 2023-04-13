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

import sift_script

#===================================================
#                   MAIN FUNCTION
#===================================================

def main(source_image_path, target_image_path):
    sift_script.sift(source_image_path, target_image_path)

if __name__ == '__main__':
    main('materials\lab_5_6\source.png', 'materials\lab_5_6\simple.png')