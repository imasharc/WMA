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

def main():
    sift_script.sift('materials\lab_5_6\saw1.jpg', 'materials\lab_5_6\saw2.jpg')

if __name__ == '__main__':
    main()