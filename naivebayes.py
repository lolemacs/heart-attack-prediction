import load
import numpy as np
from sklearn import naive_bayes
from sklearn.metrics import f1_score, accuracy_score

trX, teX, trY, teY = load.loadData(onehot = False, poly = 3, prep = 'std')

gnb = naive_bayes.GaussianNB()
gnb.fit(trX, trY)
print "Training F1: ", f1_score(trY, gnb.predict(trX))
print 'Test F1:', f1_score(teY, gnb.predict(teX))
