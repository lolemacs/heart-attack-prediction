import xgboost as xgb
import load
from sklearn.metrics import f1_score, accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

def modelfit(alg, trX, trY, teX, teY ,useTrainCV=True, cv_folds=5, early_stopping_rounds=35):
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
    print "Accuracy : %.4g" % metrics.accuracy_score(trY, dtrain_predictions)
    print "Test Accuracy : %.4g" % metrics.accuracy_score(teY, alg.predict(teX))

trX, teX, trY, teY = load.loadData(onehot = False)
"""
xgb1 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=2,
 min_child_weight=3,
 gamma=0.1,
 reg_alpha=1.0,
 subsample=0.7,
 colsample_bytree=0.7,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1)

modelfit(xgb1, trX, trY, teX, teY)

quit()



param_test6 = {
 'reg_alpha':[0.8, 1, 1.2]
}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=81, max_depth=2,
 min_child_weight=3, gamma=0.1, subsample=0.7, colsample_bytree=0.7,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(trX,trY)
print gsearch1.best_params_, gsearch1.best_score_


quit()
"""









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
