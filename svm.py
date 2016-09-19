import load
import numpy as np
from sklearn import svm as sksvm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.grid_search import GridSearchCV

trX, teX, trY, teY = load.loadData(onehot = False, prep = 'std')


"""
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
"""


param = {
 'kernel':['poly'],
 'C':[0.01, 0.1, 1.0, 10.0],
 'degree':[1, 2, 3, 4, 5]
}

gsearch = GridSearchCV(estimator = sksvm.SVC(class_weight = 'balanced'), 
 param_grid = param,n_jobs=4,iid=False, cv=5, scoring='f1')

gsearch.fit(trX,trY)

print gsearch.best_params_, gsearch.best_score_

print "Training F1: ", f1_score(trY, gsearch.predict(trX))
print 'Test F1:', f1_score(teY, gsearch.predict(teX))
