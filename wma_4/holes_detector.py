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

class Blurr:

    def __init__(self, kernel_size, neumann_neighbourhood = False):
        self.kernel = np.ones((kernel_size, kernel_size))
        if neumann_neighbourhood:
            neighbourhood_radius = kernel_size // 2
            center = neumann_neighbourhood + 1
            for y in range(kernel_size):
                for x in range(kernel_size):
                    if abs(center - x) + abs(center - y) > neighbourhood_radius:
                        self.kernel[y, x] = 0
        print(self.kernel)

class EdgeDetector:

    def __init__(self):
        pass

class CircleDetector:

    def __init__(self, min_distance_between_centers, accumulator_threshold, min_radius = 0, max_radius = 0):
        self.min_distance_between_centers = min_distance_between_centers
        self.accumulator_threshold = accumulator_threshold
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect(self, image):
        pass

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
    Blurr(5, True)
    # main_window = GUI_MainWindow()
    # main_window.run()

if __name__ == '__main__':
    main()