# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans

def trainmodel(features):
    def getFeatVec(features,clf):
        featVec = np.zeros((1, 1000))
        res = clf.predict(features)
        for i in res:
            featVec[0][i] += 1
        return featVec
    def learnVocabulary(dataset):
        features = np.vstack(dataset)
        print( features.shape)
        clf = MiniBatchKMeans(n_clusters=1000,init_size=3000,verbose=False)
        clf.fit(features)
        joblib.dump(clf , "./05vocab.pkl")
        print( "Done")
        return clf
    def trainClassifier(dataset,tags,clf=None):
        trainData = np.float32([]).reshape(0, 1000)
        response = np.int64([])
        if clf == None:
            clf = joblib.load("./05vocab.pkl")
        target =  {'sheep': 0, 'elk': 1, 'horse': 2, 'koala': 3, 'bicycle': 4,
            'monkey': 5, 'cow': 6, 'giraffe': 7, 'whale': 8, 'car': 9, 'fox': 10,
         'plane': 11, 'tiger': 12, 'lion': 13, 'train': 14, 'bear': 15,
          'statue': 16, 'tower': 17, 'puppy': 18, 'bird': 19, 'zebra': 20}
        for i in dataset:
            featVec = getFeatVec(i, clf)
            trainData = np.append(trainData, featVec, axis=0)

        print( "Now train svm classifier...")
        print( trainData.shape,len(tags))
        #use sklearn svm
        clf = svm.SVC(kernel='linear',C=1,probability=True)
        clf.fit(trainData,tags)
        joblib.dump(clf, "./05svm.model")
        print( "Done")
    feature = features[0]
    tags = features[1]
    print('start kmeans')
    clf = learnVocabulary(feature)
    print('start svm')
    trainClassifier(feature,tags,clf)
    print('done')
