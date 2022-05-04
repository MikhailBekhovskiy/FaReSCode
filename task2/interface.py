from tkinter import *
from tkinter import ttk
from random import randint
import pandas as pd
from PIL import ImageTk, Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy import fftpack
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import collections

class App:
    def __init__(self, root):
        # initializing main frame
        self.mainframe = ttk.Frame(root, width = 300, height = 300)
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # internal variables
        self.bases = ['ORL', 'ORL_cloak']
        self.db = StringVar()
        self.db.set(self.bases[0])
        root.title(f'FaReS Modeling with {self.db.get()}')
        self.methods = ['histograms', 'scaling', 'dft', 'dct', 'sliding_gradient']
        self.method = StringVar()
        self.method.set(self.methods[0])
        self.dynamic = None
        self.data = None
        self.mod = None
        self.features = None
        self.final_fig = None
        self.DATA_CONF = {
                            'ORL':{
                                'img_path':'./data/ORL/s{g}/{im}.png'
                            },
                            'ORL_cloak':{
                                'img_path':'./data/ORL_cloak/s{g}/{im}_cloaked.jpg'
                            },
                            'ORL_mask':{
                                'img_path':'./data/ORL_mask/s{g}/{im}-with-mask.jpg'
                            }
                        }
        self.best_parametres = {
            'histograms':StringVar(),
            'scaling': StringVar(),
            'dft':StringVar(),
            'dct':StringVar(),
            'sliding_gradient':StringVar()
        }
        self.table = None
        self.best_parametres['histograms'].set(30)
        self.best_parametres['scaling'].set(0.5)
        self.best_parametres['dft'].set(4)
        self.best_parametres['dct'].set(6)
        self.best_parametres['sliding_gradient'].set(10)
        self.parametres = None
        self.parFirst = StringVar()
        self.parLast = StringVar()
        self.parStep = StringVar()
        base = 'ORL'
        self.data = self.load(base)
        OptionMenu(self.mainframe, self.method, *self.methods).grid(column = 0, row = 0)
        self.Im = ttk.Label(self.mainframe, text='Dynamic demonstration',compound='top')
        self.Im.grid(column=6, row=0, columnspan=4, rowspan=3)
        ttk.Label(self.mainframe, text='first parametre value').grid(column=0, row = 1)
        ttk.Entry(self.mainframe, width=7, textvariable=self.parFirst).grid(column=1, row=1)

        ttk.Label(self.mainframe, text='last parametre value').grid(column=2, row=1)
        ttk.Entry(self.mainframe, width=7, textvariable=self.parLast).grid(column=3, row=1)

        ttk.Label(self.mainframe, text='step size').grid(column=4, row=1)
        ttk.Entry(self.mainframe, width=7, textvariable=self.parStep).grid(column=5, row=1)
        # print(self.data['target'])
        ttk.Label(self.mainframe, text='hist parametre').grid(column=0, row=3)
        ttk.Entry(self.mainframe, width=7, textvariable=self.best_parametres['histograms']).grid(column=1, row=3)
        ttk.Label(self.mainframe, text='scaling parametre').grid(column=2, row=3)
        ttk.Entry(self.mainframe, width=7, textvariable=self.best_parametres['scaling']).grid(column=3, row=3)
        ttk.Label(self.mainframe, text='dft parametre').grid(column=4, row=3)
        ttk.Entry(self.mainframe, width=7, textvariable=self.best_parametres['dft']).grid(column=5, row=3)
        ttk.Label(self.mainframe, text='dct parametre').grid(column=6, row=3)
        ttk.Entry(self.mainframe, width=7, textvariable=self.best_parametres['dct']).grid(column=7, row=3)
        ttk.Label(self.mainframe, text='gradient parametre').grid(column=8, row=3)
        ttk.Entry(self.mainframe, width=7, textvariable=self.best_parametres['sliding_gradient']).grid(column=9, row=3)
        ttk.Button(self.mainframe, text='run', command=self.research_s).grid(column=0, row=2)
        ttk.Button(self.mainframe, text='default parametres', command=self.set_def).grid(column=2, row=2)
        ttk.Button(self.mainframe, text='Parallel research', command=self.research_p).grid(column=0, row=4)
        OptionMenu(self.mainframe, self.db, *self.bases).grid(column=1, row=4)
        ttk.Button(self.mainframe, text='set base', command=self.set_base).grid(column=2, row=4)
        

    # utilities
    def load(self, database):
        db_data = {'images': [], 'target':[]}
        for i in range(1, 41):
            for j in range(1, 11):
                img = cv2.imread(self.DATA_CONF[database]['img_path'].format(g=i, im=j), -1)
                db_data['images'].append(img)
                db_data['target'].append(i)
        return db_data

    def update(self):
        self.Im['image'] = self.dynamic
    
    def set_def(self):
        method = self.method.get()
        if method == 'scaling':
            self.parFirst.set('0.5')
            self.parLast.set('1')
            self.parStep.set('0.1')
        elif method in ['dft', 'dct', 'sliding_gradient']:
            self.parFirst.set('2')
            self.parLast.set('20')
            self.parStep.set('2')
        else:
            self.parFirst.set('5')
            self.parLast.set('50')
            self.parStep.set('5')

    def set_base(self):
        base = self.db.get()
        root.title(f'FaReS Modeling with {base}')
        self.data = self.load(base)

    # for demonstration
    def show(self, data, num_photo):
        fig, ((ax1), (ax2)) = plt.subplots(1,2, figsize = (15,6))
        ax2.imshow(self.features[num_photo], cmap=plt.cm.gray)
        ax1.imshow(data[num_photo], cmap=plt.cm.gray)
        # plt.show()
        fig.savefig('results/demonstr.png')
        plt.close(fig)

    def plot(self, data, num_photo):
        fig, ((ax1), (ax4)) = plt.subplots(1,2, figsize = (15,6))
        ax4.plot(self.features[num_photo])
        ax1.imshow(self.data['images'][num_photo], cmap=plt.cm.gray)
        # plt.show()
        fig.savefig('results/demonstr.png')
        plt.close(fig)

    def hist(self, data, num_photo):
        fig, ((ax1), (ax4)) = plt.subplots(1,2, figsize = (15,6))
        ax4.hist(self.features[num_photo])
        ax1.imshow(self.data['images'][num_photo], cmap=plt.cm.gray)
        # plt.show()
        fig.savefig('results/demonstr.png')
        plt.close(fig)


    def plot3d(self, array_acc, best_params, best_score):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.array(array_acc)[:,1], np.array(array_acc)[:, 0], np.array(array_acc)[:, 2], color='red', depthshade=False)
        ax.set_xlabel(f'Test size\nBest result {best_score}\nWith parametre {best_params[0]}\nTest size {best_params[1]}')
        ax.set_ylabel('Parametre value')
        ax.set_zlabel('Accuracy')
        fig.set_figwidth(9)
        fig.set_figheight(9)
        return fig

    def plot_par(self, full_acc):
        x = np.array([1,2,3,4,5,6,7,8,9])
        y = [0 for i in [1,2,3,4,5,6,7,8,9]]
        y1 = [4 for i in [1,2,3,4,5,6,7,8,9]]
        y2 = [2 for i in [1,2,3,4,5,6,7,8,9]]
        y3 = [3 for i in [1,2,3,4,5,6,7,8,9]]
        y4 = [1 for i in [1,2,3,4,5,6,7,8,9]]
        y5 = [5 for i in [1,2,3,4,5,6,7,8,9]]

        z = full_acc[:,0]
        z1 = full_acc[:,1]
        z2 =  full_acc[:, 2]
        z3 = full_acc[:,3]
        z4 = full_acc[:,4]
        z5 = full_acc[:,5]

        fig = plt.figure()
        fig.set_figwidth(9)
        fig.set_figheight(9)

        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, label='Histograms')
        ax.plot(x, y1, z1, label='Scale')
        ax.plot(x, y2, z2, label='DFT')
        ax.plot(x, y3, z3, label='DCT')
        ax.plot(x, y4, z4, label='Gradient')
        ax.plot(x, y5, z5, label='parallel')

        ax.set_xlabel('Test size')
        ax.set_ylabel('Method')
        ax.set_zlabel('Accuracy')

        ax.legend()
        return fig

    # feature extraction
    def histograms(self, data, bins):
        histed_data = []
        for img in data:
            hist = np.histogram(img, int(bins))
            histed_data.append(hist[0])
        return np.array(histed_data)

    def scaling(self, data, scale):
        scaled_data = []
        for img in data:
            shape = img.shape[0]
            width = int(shape * scale)
            dim = (width, width)
            scaled_data.append(cv2.resize(img, dim))
        return np.array(scaled_data)

    def dft(self, data, matrix_size):
        dft_data = []
        for img in data:
            dft = np.fft.fft2(img)
            dft = np.real(dft)
            dft_data.append(dft[:int(matrix_size), :int(matrix_size)])
        return np.array(dft_data)

    def dct(self, data, matrix_size):
        dct_data = []
        for img in data:
            dct = scipy.fftpack.dct(img, axis=1)
            dct = scipy.fftpack.dct(dct, axis=0)
            dct_data.append(dct[:int(matrix_size), :int(matrix_size)])
        return np.array(dct_data)

    def sliding_gradient(self, data, height):
        gradients = []
        h = int(height)
        for img in data:
            shape = img.shape[0]
            i = 1
            result = []
            while i * h + 2 * h <= shape:
                prev = np.array(img[i*h:i*h + h, :])
                nxt = np.array(img[i * h + h: i * h + 2 * h, :])
                result.append(prev - nxt)
                i += 1
            result = np.array(result)
            result = result.reshape((result.shape[0] * result.shape[1], result.shape[2]))
            result = np.mean(result, axis=0)
            gradients.append(result)
        return np.array(gradients)

    # main computations
    def research_s(self):
        parpar = (float(self.parFirst.get()), float(self.parLast.get()), float(self.parStep.get()))
        im_num = randint(0, 399)
        cur = parpar[0] + parpar[2]
        self.parametres = [parpar[0]]
        while cur <= parpar[1]:
            self.parametres.append(cur)
            cur += parpar[2]

        best_score = 0.
        params_acc = []
        meth = self.method.get()
        for param in self.parametres:
            method = eval(f'self.{meth}')
            self.features = method(self.data['images'], param)
            X = self.features
            Y = self.data['target']
            classifier = KNeighborsClassifier(n_neighbors=1)
            for t_size in range(1, 10):
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=t_size * 0.1, stratify=Y, random_state=24)
                X_train = X_train.reshape(X_train.shape[0], -1)
                classifier.fit(X_train, Y_train)
                X_test = X_test.reshape(X_test.shape[0], -1)
                Y_predicted = classifier.predict(X_test)
                final_acc = accuracy_score(Y_predicted, Y_test)
                params_acc.append([param, t_size, final_acc])
                if final_acc > best_score:
                    best_params = [param, t_size]
                    best_score = final_acc
        fig = self.plot3d(params_acc, best_params, best_score).suptitle(meth)
        plt.savefig(f'results/{meth}.png')
        plt.show()
        plt.close()
        self.best_parametres[meth].set(best_params[0])
        return best_params, best_score, params_acc

    def get_predict(self, method, size):
        classifier = KNeighborsClassifier(n_neighbors=1)
        X = eval(f'self.{method}')(self.data['images'], float(self.best_parametres[method].get()))
        y = self.data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size * 0.1, stratify=y, random_state=24)
        X_train = X_train.reshape(X_train.shape[0],-1)
        classifier.fit(X_train, y_train)
        X_test = X_test.reshape(X_test.shape[0],-1)
        y_predicted = classifier.predict(X_test)
        return y_predicted, y_test

    def research_p(self):
        full_acc = []
        for size in range(1, 10):
            array_cls_acc=[]
            array_y_pred = []
            par_system_y_pred = []
            for method in self.methods:
                y_pred, y_test = self.get_predict(method, size)
                array_cls_acc.append(float('%.3f'%(accuracy_score(y_pred, y_test))))
                array_y_pred.append(y_pred)
            array_y_pred = np.array(array_y_pred)
            for j in range(array_y_pred.shape[1]):
                par_system_y_pred.append(collections.Counter(array_y_pred[:,j]).most_common(1)[0][0])
            array_cls_acc.append(float('%.3f'%(accuracy_score(par_system_y_pred, y_test))))
            full_acc.append(array_cls_acc)
        fig = self.plot_par(np.array(full_acc))
        plt.savefig('results/parallel.png')
        plt.show()
        plt.close()
        self.table = pd.DataFrame(full_acc, columns=['Histograms', 'Scaling', 'DFT', 'DCT', 'Gradients', 'Parallel'])
        self.table.index = ['9/1 | 360/40','8/2 | 320/80','7/3 | 280/120','6/4 | 240/160','5/5 | 200/200','4/6 | 160/240','3/7 | 120/280','2/8 | 80/320','1/9 | 40/360']
        cols = list(self.table.columns)

        tree = ttk.Treeview(self.mainframe)
        tree.grid(column=0, row=5, columnspan=7)
        tree['columns'] = cols
        for i in cols:
            tree.column(i, width=100, anchor='w')
            tree.heading(i, text=i, anchor='w')
        for index, row in self.table.iterrows():
            tree.insert('', 0, text=index, values = list(row))

        # tree.grid(column=0, row=6)
        return np.array(full_acc)




root = Tk()
App(root)
root.mainloop()
