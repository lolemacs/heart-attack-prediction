import load
import numpy as np
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score

trX, teX, trY, teY = load.loadData(onehot = False, poly = 5, prep = 'std')

dt = tree.DecisionTreeClassifier(max_depth = 6)
dt.fit(trX, trY)
print "Training F1: ", f1_score(trY, dt.predict(trX))
print 'Test F1:', f1_score(teY, dt.predict(teX))
