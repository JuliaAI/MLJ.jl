module TestXGBoost

using MLJ
using Test
using Pkg
Pkg.clone("https://github.com/dmlc/XGBoost.jl")
import XGBoost


train_X, train_Y = readlibsvm("../data/agaricus.txt.train", (6513, 126))
test_X, test_Y = readlibsvm("../data/agaricus.txt.test", (1611, 126))

dtrain = XGBoost.DMatrix("../data/agaricus.txt.train")
dtest = XGBoost.DMatrix("../data/agaricus.txt.test")




#import XGBoost_
bareboost = XGBoostRegressor(num_round=6)
bfit,= MLJ.fit(bareboost,1,train_X,train_Y)


num_round=6
bst = XGBoost.xgboost(train_X, num_round, label = train_Y)
bst_pred = XGBoost.predict(bst,test_X)
bst_DMatrix, = MLJ.fit(bareboost,1,dtrain)




bpredict_fulltree = MLJ.predict(bareboost,bfit,test_X)
bpredict_onetree = MLJ.predict(bareboost,bfit,test_X,ntree_limit=1)
bpredict_DMatrix = MLJ.predict(bareboost,bst_DMatrix,dtest)



@test  bpredict_fulltree !=bpredict_onetree
@test  bpredict_fulltree ≈ bpredict_DMatrix atol=0.000001
@test_logs (:warn,"updater has been changed to shotgun, the default option for booster=\"gblinear\"") XGBoostRegressor(num_round=1,booster="gblinear",updater="grow_colmaker")
@test_logs (:warn,"\n num_class has been changed to 2") XGBoostClassifier(eval_metric="mlogloss",objective="multi:softprob")
@test bst_pred ≈ bpredict_fulltree atol=0.000001
end
true
