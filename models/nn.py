import load
import numpy as np
from sklearn import neighbors
from sklearn.metrics import f1_score, accuracy_score
from sklearn.grid_search import GridSearchCV

trX, teX, trY, teY = load.loadData(onehot = False, poly = 1, prep = 'std')

param = {
 'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8],
 'weights':['uniform','distance'],
 'p':[1,2]
}

gsearch = GridSearchCV(estimator = neighbors.KNeighborsClassifier(), 
 param_grid = param,n_jobs=4,iid=False, cv=5, scoring='f1')

gsearch.fit(trX,trY)

print gsearch.best_params_, gsearch.best_score_

print "Training F1: ", f1_score(trY, gsearch.predict(trX))
print 'Test F1:', f1_score(teY, gsearch.predict(teX))
