import xgboost as xgb
import load
from sklearn.metrics import f1_score, accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

def modelfit(alg, trX, trY, teX, teY ,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    dtrain = xgb.DMatrix(trX, label=trY)
    dtest = xgb.DMatrix(teX, label=teY)

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(trX, label=trY)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=1)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(trX, trY, eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(trX)
    dtrain_predprob = alg.predict_proba(trX)[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "f1 : %.4g" % f1_score(trY, dtrain_predictions)
    print "Test f1 : %.4g" % f1_score(teY, alg.predict(teX))

trX, teX, trY, teY = load.loadData(onehot = False, poly=3)


"""
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=500,
 max_depth=4,
 min_child_weight=3,
 gamma=0.,
 subsample=0.7,
 colsample_bytree=0.7,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=100.0)

modelfit(xgb1, trX, trY, teX, teY)

quit()
"""

param_test = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

gsearch = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=135, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.7, colsample_bytree=0.7,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=100.0,seed=27), 
 param_grid = param_test, scoring='f1',n_jobs=4,iid=False, cv=5)

gsearch.fit(trX,trY)
print gsearch.best_params_, gsearch.best_score_

print "Training F1: ", f1_score(trY, gsearch.predict(trX))
print 'Test F1:', f1_score(teY, gsearch.predict(teX))


quit()










dtrain = xgb.DMatrix(trX, label=trY)
dtest = xgb.DMatrix(teX, label=teY)

def predict(x):
    p = bst.predict(x)
    p = [0 if x < 0.5 else 1 for x in p]
    return p

param = {'max_depth':2, 'min_child_weight':3, 'eta':1, 'silent':1, 'objective':'binary:logistic', "scale_pos_weight":1.0, "min_child_weight":3.0, 'subsample':0.7, 'colsample_bytree':0.7,
        'gamma':0.1, 'reg_alpha':1.0}
num_round = 2000
evallist  = [(dtest,'eval'), (dtrain,'train')]
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=35)
preds = bst.predict(dtest)

print "Accuracy : %.4g" % metrics.accuracy_score(trY, predict(dtrain))
print "Test Accuracy : %.4g" % metrics.accuracy_score(teY, predict(dtest))

print "Training F1: ", f1_score(trY, predict(dtrain))
print 'Test F1:', f1_score(teY, predict(dtest))
