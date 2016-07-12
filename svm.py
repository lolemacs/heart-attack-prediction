import load
import numpy as np
from sklearn import svm as sksvm
from sklearn.metrics import f1_score, accuracy_score

trX, teX, trY, teY = load.loadData(onehot = False, poly = 3, prep = 'std')

svm = sksvm.SVC(kernel='linear', class_weight = 'auto', C = 1)
svm.fit(trX, trY)
print "----- Linear Kernel -----"
print "Training F1: ", f1_score(trY, svm.predict(trX))
print 'Test F1:', f1_score(teY, svm.predict(teX))

print "----- RBF Kernel -----"
svm = sksvm.SVC(kernel='rbf', class_weight = 'auto', C = 1)
svm.fit(trX, trY)
print "Training F1: ", f1_score(trY, svm.predict(trX))
print 'Test F1:', f1_score(teY, svm.predict(teX))
