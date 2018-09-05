#!/usr/bin/python
#coding:utf-8

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

VGG_MEAN = [103.939, 116.779, 123.68]
mpl.rcParams['font.sans-serif']=['SimHei'] # 正常显示中文标签
mpl.rcParams['axes.unicode_minus']=False # 正常显示正负号

def load_image(path):
    fig = plt.figure("Before & After")
    
    img = io.imread(path)
    img = img / 255.0
    
    ax0 = fig.add_subplot(131)
    ax0.set_xlabel(u'Original Picture')
    ax0.imshow(img)
    
    short_edge = min(img.shape[:2])
    y = (img.shape[0] - short_edge) / 2
    x = (img.shape[1] - short_edge) / 2
    crop_img = img[y:y+short_edge, x:x+short_edge]
    
    ax1 = fig.add_subplot(132)
    ax1.set_xlabel(u"Centre Picture")
    ax1.imshow(crop_img)
    
    re_img = transform.resize(crop_img, (224, 224))
    
    ax2 = fig.add_subplot(133)
    ax2.set_xlabel(u"Target Picture")
    ax2.imshow(re_img)
    return re_img

def percent(value):
    return '%.2f%%' % (value * 100)

if __name__ == "__main__":
    test()
