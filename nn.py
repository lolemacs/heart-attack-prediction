import load
import numpy as np
from sklearn import neighbors
from sklearn.metrics import f1_score, accuracy_score

n_neighbors = 3

trX, teX, trY, teY = load.loadData(onehot = False, poly = 10, prep = 'std')

for weights in ['uniform', 'distance']:
    nn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    nn.fit(trX, trY)
    print "----- %s weights -----" % weights
    print "Training F1: ", f1_score(trY, nn.predict(trX))
    print 'Test F1:', f1_score(teY, nn.predict(teX))
