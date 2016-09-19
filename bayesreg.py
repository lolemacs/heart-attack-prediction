import load
import numpy as np
from sklearn import linear_model
from sklearn.metrics import f1_score, accuracy_score
from sklearn.grid_search import GridSearchCV

trX, teX, trY, teY = load.loadData(onehot = False, prep = 'std', poly = 3)

def predict(x):
    p = gsearch.predict(x)
    p = [0 if x < 0.5 else 1 for x in p]
    return p

param = {
 'alpha_1':[1e-2, 0.1, 1.0, 10., 100., 1000., 10000.],
 'alpha_2':[1e-12],
 'lambda_1':[1e-13],
 'lambda_2':[1e-4, 1e-2, 0.1, 1.0]
}

gsearch = GridSearchCV(estimator = linear_model.BayesianRidge(), 
 param_grid = param,n_jobs=4,iid=False, cv=5)

gsearch.fit(trX,trY)

print gsearch.best_params_, gsearch.best_score_

print "Training F1: ", f1_score(trY, predict(trX))
print 'Test F1:', f1_score(teY, predict(teX))
