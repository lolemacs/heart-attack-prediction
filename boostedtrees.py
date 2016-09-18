import xgboost as xgb
import load
from sklearn.metrics import f1_score, accuracy_score

trX, teX, trY, teY = load.loadData(onehot = False)

dtrain = xgb.DMatrix(trX, label=trY)
dtest = xgb.DMatrix(teX, label=teY)

def predict(x):
    p = bst.predict(x)
    p = [0 if x < 0.5 else 1 for x in p]
    return p

param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic', "scale_pos_weight":1.0, "min_child_weight":10.0 }
num_round = 2000
evallist  = [(dtest,'eval'), (dtrain,'train')]
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)
preds = bst.predict(dtest)

print "Training F1: ", f1_score(trY, predict(dtrain))
print 'Test F1:', f1_score(teY, predict(dtest))
