# -*- coding: utf-8 -*-

import csv
import os
import shutil
import random
from sklearn.model_selection import train_test_split

def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def save_csv(name,set):
    with open(name,"w") as datacsv:
        csvwriter = csv.writer(datacsv)
        csvwriter.writerows(set)

rootdir="new_data"
file = []
dir = []
whole = {}
for parent,dirnames,filenames in os.walk(rootdir):
    for filename in filenames:
        try:
            paths = parent.split("/")
            iclass = paths[1]
            ipath = os.path.join(parent,filename)
            if iclass not in dir:
                dir.append(iclass)
                whole[iclass]=[]
            file.append([iclass,ipath])
        except Exception as e:
            pass

for i in file:
    whole[i[0]].append(i)

train = []
test = []
for i in whole:
    a,b = train_test_split(whole[i],test_size = 0.2)
    train+=a
    test+=b


mkdir("train")
mkdir("test")
for i in dir:
    mkdir("train/"+i)
    mkdir("test/"+i)

for i in train:
    shutil.copy(i[1],"train/"+i[0]+"/"+i[1].split("/")[2])
for i in test:
    shutil.copy(i[1],"test/"+i[0]+"/"+i[1].split("/")[2])
