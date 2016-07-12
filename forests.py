import load
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

trX, teX, trY, teY = load.loadData(onehot = False, poly = 10, prep = 'std')

sample_weight = load.get_weights(trY)

rf = RandomForestClassifier(n_estimators=100, max_features=3)

rf.fit(trX, trY, sample_weight = sample_weight)
print "Training F1: ", f1_score(trY, rf.predict(trX))
print 'Test F1:', f1_score(teY, rf.predict(teX))
