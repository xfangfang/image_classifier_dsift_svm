from load import loaddata
from feature import extractfeature
from classify import trainmodel
from predict import testmodel

if __name__ == '__main__':
    data,tags = loaddata('./train')
    features = extractfeature(data,tags)
    trainmodel(features)
    res = testmodel('./test')
