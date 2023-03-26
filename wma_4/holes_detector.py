#===================================================
#                       IMPORTS
#===================================================

import cv2
import numpy as np
import tkinter as tk # Python GUI library
import tkinter.filedialog # This must be imported explicitly to use filedialog
from PIL import Image
from PIL import ImageTk

#===================================================
#                   CIRCLE DETECTION
#===================================================

class CircleDetector:

    def __init__(self, gradient_variant, dp, min_distance, param_1 = 60, param_2 = 40, min_radius = 0, max_radius = 0):
        self.gradient_variant = gradient_variant
        self.dp = dp
        self.min_distance = min_distance
        self.param_1 = param_1
        self.param_2 = param_2
        self.min_radius = min_radius
        self. max_radius = max_radius

    def detect(self, image):
        circles = cv2.HoughCircles(image, self.gradient_variant, self.dp, self.min_distance,
                                param1 = self.param_1, param2 = self.param_2,
                                minRadius = self.min_radius, maxRadius = self.max_radius)
        return np.uint16(np.around(circles))

#===================================================
#                         GUI
#===================================================

class GUI_MainWindow:
    def __init__(self, circle_detector, refresh_rate = 33):
        self.root = tk.Tk()

        self.refresh_rate = refresh_rate
        self.circle_detector = circle_detector

        self.video = None
        self.panel_a = None

        self.btn = tk.Button(self.root, text = 'Select a video', command = self.load_video)
        self.btn.pack(side = 'bottom', fill = 'both', expand = 'yes', padx = '10', pady = '10')

    def load_video(self):
        path = tk.filedialog.askopenfile()
        self.video = cv2.VideoCapture(path.name)
        print(f'Path to the file is {path.name}')
        # self.root.after(33, self.update)
        self.update()
    
    def read_frame_from_video(self):
        if self.video is None:
            return None
        
        ret, frame = self.video.read()
        if not ret:
            return None
        return frame
        
    def update_video_display(self, frame):

        frame = cv2.resize(frame, (360, 360))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = self.circle_detector.detect(gray)
        for i in circles[0,:]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((800, 600), Image.Resampling.LANCZOS)
        image = ImageTk.PhotoImage(image)

        if self.panel_a is None:
            self.panel_a = tk.Label(image = image)
            self.panel_a.image = image
            self.panel_a.pack(side = 'left', padx = 10, pady = 10)
            print(circles)
        else:
            self.panel_a.configure(image = image)
            self.panel_a.image = image

    def update(self):
        frame = self.read_frame_from_video()
        if frame is not None:
            self.update_video_display(frame)
        self.root.after(self.refresh_rate, self.update)

    def run(self):
        self.root.mainloop()

#===================================================
#                   MAIN FUNCTION
#===================================================

def main():
    circle_detector = CircleDetector(cv2.HOUGH_GRADIENT, 1, 20, param_1 = 20, param_2 = 80)
    main_window = GUI_MainWindow(circle_detector)
    main_window.run()

if __name__ == '__main__':
    main()