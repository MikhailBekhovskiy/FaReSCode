import cv2
import numpy as numpy
from matplotlib import pyplot as plt 
from matplotlib.pyplot import figure
from tkinter import *
from tkinter import ttk, filedialog
from PIL import ImageTk, Image

class App:
    def __init__(self, root):
        root.title('Viola-Jones detector')
        mainframe = ttk.Frame(root)
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.image = None
        self.result = None
        self.to_show_im = PhotoImage()
        self.to_show_res = PhotoImage()

        self.Im = ttk.Label(mainframe, text='Photo', compound='bottom')
        self.Im.grid(column=0, row=0)
        self.Res = ttk.Label(mainframe, text = 'Result', compound='bottom')
        self.Res.grid(column=1, row=0)

        ttk.Button(mainframe, text='Choose photo', command=self.get_photo).grid(column=0, row=1)
        ttk.Button(mainframe, text='Detect face', command=self.vj_matcher).grid(column=1, row=1)

    def get_photo(self):
        path = filedialog.askopenfilename()
        self.image = cv2.imread(path)
        img = Image.open(path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        self.to_show_im=img
        self.Im['image'] = self.to_show_im

    def vj_matcher(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        line_width = 3
        face_color=(0, 0, 255)
        eyes_color=(0, 255, 0)
        scale_factor = 1.3
        min_neighbours = 5
        img = self.image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbours)

        for x, y, w, h in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), face_color,
                                line_width)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh),
                            eyes_color, line_width)

        self.result = img
        cv2.imwrite('result_vj.jpg', img)
        img = Image.open('result_vj.jpg')
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        self.to_show_res = img
        self.Res['image'] = self.to_show_res


root = Tk()
App(root)
root.mainloop()
