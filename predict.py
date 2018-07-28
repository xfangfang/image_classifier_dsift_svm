# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans
from load import loaddata
from feature import extractfeature

def testmodel(path):
    def getFeatVec(features,clf):
        featVec = np.zeros((1, 1000))
        res = clf.predict(features)
        for i in res:
            featVec[0][i] += 1
        return featVec

    def result(predict_y,test_y,num):
        print("###############################")
        right = 0
        res = []
        for i,tag in enumerate(test_y):
            if tag in predict_y[i][21-num:21]:
                right += 1
                res.append(True)
            else:
                res.append(False)
        print( "%d/%d = %f"%(right,len(test_y),right*1.0/len(test_y)))
        print("###############################")
        return res

    print('load model')
    svm = joblib.load("./05svm.model")
    clf = joblib.load("./05vocab.pkl")
    data,tags = loaddata(path)
    features,tags = extractfeature(data,tags)

    print('predict')
    test_x = np.float32([]).reshape(0,1000)
    for feature in features:
        featVec = getFeatVec(feature, clf)
        test_x = np.append(test_x,featVec,axis=0)
    p = svm.predict_proba(test_x)
    p = p.argsort()
    res = result(p,tags,5)
    return res
