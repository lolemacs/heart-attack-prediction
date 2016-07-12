import cPickle
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

def standardize(X):
    return preprocessing.StandardScaler().fit_transform(X)

def minmax(X):
    return preprocessing.MinMaxScaler().fit_transform(X)

prepFuncs = {"std" : standardize, "mm" : minmax}

def polyExpand(X, order):
    return PolynomialFeatures(order).fit_transform(X)

def one_hot(x,n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h

def loadData(onehot = True, prep = None, poly = None):
    with open("data.pkl","rb") as f:
        data = cPickle.load(f)
    X = data["data"].astype('float64')
    Y = data["labels"]
    
    if prep: X = prepFuncs[prep](X)
    if poly != None: X = polyExpand(X, poly)
    if onehot: Y = one_hot(Y,2)
    
    nPoints = len(Y)
    valSplit = int(0.8*nPoints)
    
    trX = X[:valSplit]
    teX = X[valSplit:]
    trY = Y[:valSplit]
    teY = Y[valSplit:]
    return trX, teX, trY, teY
