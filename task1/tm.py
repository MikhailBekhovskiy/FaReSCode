import cv2
import numpy as numpy
from matplotlib import pyplot as plt 
from matplotlib.pyplot import figure
from tkinter import *
from tkinter import ttk, filedialog
from PIL import ImageTk, Image

class App:
    def __init__(self, root):
        # window init
        root.title('Template Matcher')
        mainframe = ttk.Frame(root)
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        # attributes for internal calculations
        self.methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        self.image = None
        self.template = None
        self.result = None
        self.method = StringVar()
        self.method.set(self.methods[0])
        # attributes for GUI display
        self.to_show_im = None
        self.to_show_tmp = None
        self.to_show_res = None
        # Labels with images
        self.Im = ttk.Label(mainframe, text='Photo', compound = 'bottom')
        self.Im.grid(column=0, row=0)
        self.Tmp = ttk.Label(mainframe, text='Template', compound = 'bottom')
        self.Tmp.grid(column=2, row=0)
        self.Res = ttk.Label(mainframe, text='Result', compound='bottom')
        self.Res.grid(column=1, row=0)
        # buttons
        ttk.Button(mainframe, text='Choose photo', command=self.get_photo).grid(column=0, row=1)
        ttk.Button(mainframe, text='Choose template', command=self.get_template).grid(column=2, row=1)
        OptionMenu(mainframe, self.method, *self.methods).grid(column=1, row=1)
        ttk.Button(mainframe, text='Match template', command=self.template_match).grid(column=1, row=2)

    def get_photo(self):
        path = filedialog.askopenfilename()
        self.image = cv2.imread(path, 0)
        img = Image.open(path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        self.to_show_im=img
        self.Im['image'] = self.to_show_im

    def get_template(self):
        path = filedialog.askopenfilename()
        self.template = cv2.imread(path, 0)
        tmp = Image.open(path)
        tmp = tmp.resize((250, 250))
        tmp = ImageTk.PhotoImage(tmp)
        self.to_show_tmp = tmp
        self.Tmp['image'] = self.to_show_tmp

    def template_match(self):
        img2 = self.image.copy()
        meth = eval(self.method.get())
        template = self.template.copy()
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img2, template, meth)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img2, top_left, bottom_right, 255, 10)
        cv2.imwrite('result.jpg', img2)
        self.result = img2
        img2 = Image.open('result.jpg')
        img2 = img2.resize((250, 250))
        img2 = ImageTk.PhotoImage(img2)
        self.to_show_res = img2
        self.Res['image'] = self.to_show_res


root = Tk()
App(root)
root.mainloop()
