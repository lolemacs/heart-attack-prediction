import load
import numpy as np
from sklearn import tree

trX, teX, trY, teY = load.loadData(onehot = False, poly = 3, stdize = False)

dt = tree.DecisionTreeClassifier(max_depth = 6, sample_weight = 'balanced')
dt.fit(trX, trY)
print "Training acc: ", dt.score(trX, trY)
print "Test acc: ", dt.score(teX, teY)
