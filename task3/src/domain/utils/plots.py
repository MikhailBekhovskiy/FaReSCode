import matplotlib.pyplot as plt
import numpy as np
import cv2
import io


def plot_descriptor(orb: np.array) -> io.BytesIO:
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(orb)
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return buf

def corners_show(kp, img):
    fig = plt.figure(figsize=(3,3))
    im = img
    im[kp>0.01*kp.max()] = [0, 0, 255]
    plt.imshow(im)
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return buf