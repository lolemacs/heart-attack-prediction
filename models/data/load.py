import cPickle
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(1234)

def standardize(X):
    return preprocessing.StandardScaler().fit_transform(X)

def minmax(X):
    return preprocessing.MinMaxScaler().fit_transform(X)

prepFuncs = {"std" : standardize, "mm" : minmax}

def polyExpand(X, order):
    return PolynomialFeatures(order).fit_transform(X)

def get_weights(Y):
    positiveWeight = np.mean(Y == 1)
    return np.array([positiveWeight if y == 1 else 1 - positiveWeight for y in Y])


def one_hot(x,n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h

def loadData(onehot = True, prep = None, poly = None, split=True, fileName='data.pkl'):
    with open("data/%s"%fileName,"rb") as f:
        data = cPickle.load(f)
    X = data["data"].astype('float64')
    Y = data["labels"]

    p = np.random.permutation(len(Y))
    X = X[p]
    Y = Y[p]
    
    if prep: X = prepFuncs[prep](X)
    if poly != None: X = polyExpand(X, poly)
    if onehot: Y = one_hot(Y,2)
    
    if split:
        nPoints = len(Y)
        valSplit = int(0.8*nPoints)
        
        trX = X[:valSplit]
        teX = X[valSplit:]
        trY = Y[:valSplit]
        teY = Y[valSplit:]
        return trX, teX, trY, teY
    else:
        return X, Y


if __name__ == "__main__":
    loadData()
