import load
import numpy as np
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score

trX, teX, trY, teY = load.loadData(onehot = False, poly = 2, prep = 'std')

sample_weight = load.get_weights(trY)

dt = tree.DecisionTreeClassifier()

dt.fit(trX, trY, sample_weight = sample_weight)
print "Training F1: ", f1_score(trY, dt.predict(trX))
print 'Test F1:', f1_score(teY, dt.predict(teX))
