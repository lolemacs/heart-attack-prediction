import load
import numpy as np
from sklearn import linear_model
from sklearn.metrics import f1_score, accuracy_score
from sklearn.grid_search import GridSearchCV

trX, teX, trY, teY = load.loadData(onehot = False, poly = 5, prep = 'std')

param = {
 'C':[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
}

gsearch = GridSearchCV(estimator = linear_model.LogisticRegression(class_weight='balanced'), 
 param_grid = param,n_jobs=4,iid=False, cv=5, scoring='f1')

gsearch.fit(trX,trY)

print gsearch.best_params_, gsearch.best_score_

print "Training F1: ", f1_score(trY, gsearch.predict(trX))
print 'Test F1:', f1_score(teY, gsearch.predict(teX))
