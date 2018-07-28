# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from scipy.misc import imresize

def loaddata(fpath='./ds2018'):
    def standarizeImage(im):
        if im.shape[0] > 300:
            resize_factor = 300.0 / im.shape[0]  # don't remove trailing .0 to avoid integer devision
            im = imresize(im, resize_factor)
        return im
    print( 'load data')
    file = []
    dir = set()
    rootdir=fpath
    for parent,dirnames,filenames in os.walk(rootdir):
        for filename in filenames:
            try:
                paths = parent.split("/")
                iclass = paths[len(paths)-1]
                dir.add(iclass)
                ipath = os.path.join(parent,filename)
                file.append([iclass,ipath])
            except Exception as e:
                pass
    data = []
    tags = []
    clss = {}
    for i,cls_str in enumerate(sorted(dir)):
        clss[cls_str] = i

    train_file = file
    for info in train_file:
        img = cv2.imread(info[1],cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = standarizeImage(img)
        data.append(img)
        tags.append(clss[info[0]])

    print('done')
    return (data,tags)
