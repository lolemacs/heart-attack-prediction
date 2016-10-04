from data import load
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.grid_search import GridSearchCV

trX, teX, trY, teY = load.loadData(onehot = False, poly = 10, prep = 'std')

sample_weight = load.get_weights(trY)


print "[+] Grid search to find best params"

param = {
 'max_features':[3,4,5,6],
 'max_depth':[2,3,4,5],
}

gsearch = GridSearchCV(estimator = RandomForestClassifier(n_estimators = 500, class_weight='balanced'), 
 param_grid = param,n_jobs=4,iid=False, cv=5, scoring='f1')

gsearch.fit(trX,trY)

print gsearch.best_params_, gsearch.best_score_

print "Training F1: ", f1_score(trY, gsearch.predict(trX))
print 'Validation F1:', f1_score(teY, gsearch.predict(teX))


print "\n\n[+] Training full model with best params"


X, Y = load.loadData(onehot = False, poly = 10, split=False, prep = 'std')

r = RandomForestClassifier(n_estimators = 500, class_weight='balanced', max_features=5, max_depth=4)
r.fit(X,Y)
print "Training F1: ", f1_score(Y, r.predict(X))

X_test, Y_test = load.loadData(onehot = False, split=False, prep='std', poly=10, fileName='test.pkl')
print 'Test F1:', f1_score(Y_test, r.predict(X_test))   
