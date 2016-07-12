import load
import numpy as np
from sklearn import linear_model
from sklearn.metrics import f1_score, accuracy_score

trX, teX, trY, teY = load.loadData(onehot = False, prep = 'std', poly = 3)

def predict(x):
    p = bayes.predict(x)
    p = [0 if x < 0.5 else 1 for x in p]
    return p

bayes = linear_model.BayesianRidge()
bayes.fit(trX, trY)

print "Training F1: ", f1_score(trY, predict(trX))
print 'Test F1:', f1_score(teY, predict(teX))
