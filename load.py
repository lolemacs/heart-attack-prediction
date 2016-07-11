import cPickle

def loadData():
    with open("data.pkl","rb") as f:
        data = cPickle.load(f)
    X = data["data"]
    Y = data["labels"]
    nPoints = len(Y)
    valSplit = int(0.8*nPoints)
    trX = X[:valSplit]
    teX = X[valSplit:]
    trY = Y[:valSplit]
    teY = Y[valSplit:]
    return trX, teX, trY, teY
