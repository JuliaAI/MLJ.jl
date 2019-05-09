# simple XGBoost example
# with tuning along the lines of this post:
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

# Warning: This is a demonstration only. The small size of the data
# set and the large number of hyperparameters being tuned implies that
# overfitting is likely.

# Declare variables const for a speed boost

# using Revise
using StatsBase
using MLJ
using Random
using Plots
plotly()

# load a task (data plus learning objective):
task = load_crabs()
shuffle!(MersenneTwister(134), task)

# list models matching the task:
models(task)

# assuming XGBoost package is in load path, we can
# load the model code for the applicable XGBoost model:
@load XGBoostClassifier


## EXTRACT A HOLDOUT TEST SET

train, test = partition(1:nrows(task), 0.7)

testtask = task[test]
task = task[train]


## BALANCED TARGET VALUES?

countmap(task.y)
# B:O is 67:73, so yes.


## DEFINE THE XGBOOST MODEL

# a time-dependent random seed is obtained with seed=-1:
boost = XGBoostClassifier(num_round=10, eta=0.05)


## DETERMINE NUMBER OF ITERATIONS FOR DEFAULT LEARNING RATE

# wrap model in task:
mach = machine(boost, task)

# range for forest.n:
r = range(boost, :num_round, lower=10, upper=1000)

# generate a learning curve, cross entropy loss vs num_round:
curve = learning_curve!(mach,
                        resampling=CV(),
                        nested_range=(num_round=r,),
                        measure=cross_entropy,
                        resolution=100)
plot(curve.parameter_values, curve.measurements)

boost.num_round = 300


## TUNE max_depth AND min_child_weight

# to check order of paramaters:
params(boost)

r1 = range(boost, :max_depth, lower=3, upper=10)
r2 = range(boost, :min_child_weight, lower=0, upper=5)

# define nested ranges, matching pattern of params(forest):
nested_ranges = (max_depth=r1, min_child_weight=r2)

tuned1 = TunedModel(model=boost,
                   tuning=Grid(resolution=8),
                   resampling=CV(),
                   nested_ranges=nested_ranges,
                   measure=cross_entropy)

# tune and plot the results:
mach = machine(tuned1, task)
fit!(mach)
plot(mach)

# fine tune:
tuned1.nested_ranges = (max_depth=range(boost, :max_depth, lower=3, upper=7),
                        min_child_weight = range(boost, :min_child_weight, lower=0, upper=2))
fit!(mach)
plot(mach)

# extract optimal model:
boost = fitted_params(mach).best_model
boost.min_child_weight # 0.57
boost.max_depth        # 3


## TUNE gamma

# We'll use a learning curve to examine the effect of gamma:
mach = machine(boost, task)
curve = learning_curve!(mach,
                        resampling=CV(),
                        nested_range=(gamma=range(boost, :gamma, lower=0, upper=10),),
                        measure=cross_entropy,
                        resolution=30)
plot(curve.parameter_values, curve.measurements)
# gamma appears to have no effect on performance
boost.gamma = 0


## RECALIBRATE NUMBER OF ITERATIONS

# generate a learning curve, cross entropy loss vs num_round:
curve = learning_curve!(mach,
                        resampling=CV(),
                        nested_range=(num_round=r,), # as before
                        measure=cross_entropy,
                        resolution=100)
plot(curve.parameter_values, curve.measurements)

boost.num_round = curve.parameter_values[argmin(curve.measurements)] # 560


## TUNE subsample AND colsample_bytree

r1 = range(boost, :subsample, lower=0.6, upper=1.0)
r2 = range(boost, :colsample_bytree, lower=0.6, upper=1.0)

# define nested ranges, matching pattern of params(forest):
nested_ranges = (subsample=r1, colsample_bytree=r2)

tuned1 = TunedModel(model=boost,
                   tuning=Grid(resolution=8),
                   resampling=CV(),
                   nested_ranges=nested_ranges,
                   measure=cross_entropy)

# tune and plot the results:
mach = machine(tuned1, task)
fit!(mach)
plot(mach)

# fine tune:
tuned1.nested_ranges = (subsample=range(boost, :subsample, lower=0.5, upper=0.85),
                        colsample_bytree=range(boost, :colsample_bytree, lower=0.8, upper=1.0))
fit!(mach)
plot(mach)

# extract optimal model:
boost = fitted_params(mach).best_model
boost.subsample          # 0.7
boost.colsample_bytree   # 1


## LOWER THE LEARNING RATE AND ADD MORE TREES

boost.eta = 0.005

# generate a learning curve, cross entropy loss vs num_round:
mach = machine(boost, task)

r = range(boost, :num_round, lower=10, upper=7000)

curve = learning_curve!(mach,
                        resampling=CV(),
                        nested_range=(num_round=r,), # as before
                        measure=cross_entropy,
                        resolution=30, verbosity=1)
plot(curve.parameter_values, curve.measurements)

boost.num_round = curve.parameter_values[argmin(curve.measurements)] # 6277


## TRAIN ON ALL THE DATA AND EVALUATE ON THE HOLDOUT SET

fit!(mach)
yhat = predict(mach, testtask.X)
cross_entropy(yhat, testtask.y)
misclassification_rate(yhat, testtask.y)



