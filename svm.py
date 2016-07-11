import load
import numpy as np
from sklearn import svm as sksvm

trX, teX, trY, teY = load.loadData(onehot = False, poly = 2, stdize = True)

svm = sksvm.SVC(kernel='linear')
svm.fit(trX, trY)
print "----- Linear Kernel -----"
print "Training acc: ", svm.score(trX, trY)
print "Test acc: ", svm.score(teX, teY)

print "----- RBF Kernel -----"
svm = sksvm.SVC(kernel='rbf')
svm.fit(trX, trY)
print "Training acc: ", svm.score(trX, trY)
print "Test acc: ", svm.score(teX, teY)
