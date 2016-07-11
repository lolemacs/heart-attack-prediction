import load
import numpy as np
from sklearn import linear_model

trX, teX, trY, teY = load.loadData(onehot = False, stdize = True)

bayes = linear_model.BayesianRidge()
bayes.fit(trX, trY)
print "Training acc: ", bayes.score(trX, trY)
print "Test acc: ", bayes.score(teX, teY)
