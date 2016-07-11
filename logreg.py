import load
import numpy as np
from sklearn import linear_model

trX, teX, trY, teY = load.loadData(onehot = False)

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(trX, trY)
print "Training acc: ", logreg.score(trX, trY)
print "Test acc: ", logreg.score(teX, teY)
