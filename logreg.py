import load
import numpy as np
from sklearn import linear_model
from sklearn.metrics import f1_score, accuracy_score

trX, teX, trY, teY = load.loadData(onehot = False, poly = 3, prep = 'std')

logreg = linear_model.LogisticRegression(C=1e5, class_weight = 'auto')
logreg.fit(trX, trY)
print "Training F1: ", f1_score(trY, logreg.predict(trX))
print 'Test F1:', f1_score(teY, logreg.predict(teX))
