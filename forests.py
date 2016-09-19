import load
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.grid_search import GridSearchCV

trX, teX, trY, teY = load.loadData(onehot = False, poly = 10, prep = 'std')

sample_weight = load.get_weights(trY)


param = {
 'max_features':[5],
 'max_depth':[4],
}

gsearch = GridSearchCV(estimator = RandomForestClassifier(n_estimators = 500, class_weight='balanced'), 
 param_grid = param,n_jobs=4,iid=False, cv=5)

gsearch.fit(trX,trY)

print gsearch.best_params_, gsearch.best_score_

print "Training F1: ", f1_score(trY, gsearch.predict(trX))
print 'Test F1:', f1_score(teY, gsearch.predict(teX))
