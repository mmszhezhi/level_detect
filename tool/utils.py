import numpy as np
import skimage
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os
import math
import glob
import scipy.misc as sm
import cv2
from keras_segmentation.predict import predict_multiple
from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
from keras_segmentation.predict import predict_multiple
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from scipy import signal
import functools


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def load_data(dir_name = 'faces_imgs'):    
    '''
    Load images from the "faces_imgs" directory
    Images are in JPG and we convert it to gray scale images
    '''
    imgs = []
    if os.path.isdir(dir_name):
        for filename in os.listdir(dir_name):
            if os.path.isfile(dir_name + '/' + filename):
                img = mpimg.imread(dir_name + '/' + filename)
                img = rgb2gray(img)
                imgs.append(img)
    elif isinstance(dir_name,list):
        for filename in dir_name:
            img = mpimg.imread(filename)
            img = rgb2gray(img)
            imgs.append(img)
    elif isinstance(dir_name,str):
        img = mpimg.imread(dir_name)
        img = rgb2gray(img)
        imgs.append(img)
    return imgs

def plot(img):
    plt.imshow(img)
    plt.show()

def visualize(imgs, format=None, gray=False):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(2, 2, plt_idx)
        plt.imshow(img, format)
    plt.show()

def load_img(path,gray_scale=False,dir=None):

    img = cv2.imread(path)
    if gray_scale:
        img_grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if dir:
        os.makedirs(dir,exist_ok=True)
        cv2.imwrite(f"{dir}/raw.png",img)
        cv2.imwrite(f"{dir}/grayscale.png",img_grayscale)
    return img

def segment(model,src,dst):
    predict_multiple(model=model,inp_dir=src,out_dir=dst)

def crop_mask_img(img,mask,crop=False):
    # img = cv2.imread(img)
    # label = cv2.imread(mask)
    label = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    # label = np.clip(label>130,0,255)
    label[label<130] = 255
    label[label < 200] = 0
    masked_img = cv2.bitwise_and(img, img, mask=label)
    if crop:
        x,y,w,h=cv2.boundingRect(cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY))
        # cv2.rectangle(masked_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        croped = cv2.getRectSubPix(masked_img,(w,h),((x+w/2),(y+h/2)))
        # cv2.imwrite(croped_out, croped)
    # cv2.imwrite(masked_out, masked_img)
    return masked_img,croped

def crop_masked_imgs(src,mask_src,dst,fixshape=(950,3850),resize=(300,1200)):
    for imgpath in os.listdir(src):
        if imgpath.endswith(".jpg"):
            try:
                img = cv2.imread(os.path.join(src,imgpath))
                mask = cv2.imread(os.path.join(mask_src,imgpath))
                label = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                label[label<130] = 255
                label[label < 200] = 0
                masked_img = cv2.bitwise_and(img, img, mask=label)
                x,y,w,h=cv2.boundingRect(cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY))
                # cv2.rectangle(masked_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                croped = cv2.getRectSubPix(masked_img,(w,h),((x+w/2),(y+h/2)))
                px,py = fixshape[0] - croped.shape[0] /2,fixshape[1] - croped.shape[1] /2
                paded = np.pad(croped,pad_width=((math.ceil(px),math.floor(px)),(math.ceil(py),math.floor(py))))
                cv2.imwrite(os.path.join(dst,imgpath), paded)
                # print(imgpath)
            except Exception as e:
                print(repr(e))

def canny_sketch(img,path=None):
    edge_img = cv2.Canny(img, 160, 80)
    # cv2.imshow("Detected Edges", edge_img)
    if path:
        cv2.imwrite(path, edge_img)
    return edge_img

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/20.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

def find_level(img):
    raw_sig = np.sum(img, axis=1)
    raw_sig = raw_sig / 1000
    raw_len = len(raw_sig)
    sig = raw_sig[int(0.2 * raw_len):int(0.8 * raw_len)]
    x = np.array(range(sig.shape[0]))
    peaks, prop = signal.find_peaks(sig, height=8, threshold=1, distance=200)
    bands = half_max_x(x[:peaks[0] + 20], sig[:peaks[0] + 20])
    round2 = functools.partial(round, ndigits=2)
    bands = [x + 0.2*raw_len for x in map(round2, bands)]
    peaks =  [x + 0.2 * raw_len for x in peaks]
    level = round(peaks[0]/len(raw_sig),2)
    return peaks,prop,bands,raw_sig,level

