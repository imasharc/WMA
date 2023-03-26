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
#                         GUI
#===================================================

class GUI_MainWindow:
    def __init__(self, refresh_rate = 33):
        self.root = tk.Tk()

        self.refresh_rate = refresh_rate

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
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((800, 600), Image.Resampling.LANCZOS)
        image = ImageTk.PhotoImage(image)

        if self.panel_a is None:
            self.panel_a = tk.Label(image = image)
            self.panel_a.image = image
            self.panel_a.pack(side = 'left', padx = 10, pady = 10)
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
    main_window = GUI_MainWindow()
    main_window.run()

if __name__ == '__main__':
    main()