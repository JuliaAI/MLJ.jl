# Simple example using MLJ ensembling to construct a "self-tuning"
# random forest.  Declare variables const for performance boost

# uncomment 2 lines for parallelized ensemble building:
# using Distributed
# addprocs()

using MLJ
using Plots
plotly()

# load a task (data plus learning objective):
task = load_boston()
models(task)

## EVALUATE A SINGLE TREE MODEL

@load DecisionTreeRegressor
tree = DecisionTreeRegressor()
mach = machine(tree, task)
evaluate!(mach, resampling=Holdout(fraction_train=0.8), measure=[rms,rmslp1])
# (MLJ.rms = 7.06, MLJ.rmslp1 = 0.33)


## DEFINE THE RANDOM FOREST MODEL

forest = EnsembleModel(atom=tree)


## GET IDEA OF NUMBER OF TREES NEEDED

# reduce number of features sampled at each nodes to sqrt of the
# number of features, a common default:
tree.n_subfeatures = 3

# construct machine:
mach = machine(forest, task)

# range for forest.n:
r = range(forest, :n, lower=10, upper=500)

# generate learning curve, rms loss vs n:
curves = learning_curve!(mach, nested_range=(n=r,), measure=rms, n=4)
plot(curves.parameter_values, curves.measurements)

forest.n = 10

                     
## DEFINE A TUNING GRID

# inspect the nested hyperparameters of our composite model:
params(forest)

# define ranges for each parameter:
r1 = range(tree, :n_subfeatures, lower=1, upper=12)
r2 = range(forest, :bagging_fraction, lower=0.4, upper=1.0)

# define nested ranges, matching pattern of params(forest):
nested_ranges = (atom=(n_subfeatures=r1,), bagging_fraction=r2)


## WRAP THE MODEL IN A TUNING STRATEGY 
# creates a new "self-tuning" model!

tuning = Grid(resolution=12)
resampling = CV()
tuned_forest = TunedModel(model=forest, tuning=tuning, resampling=resampling,
                          nested_ranges=nested_ranges, measure=rms)


## EVALUATE THE SELF-TUNING RANDOM FOREST MODEL

mach = machine(tuned_forest, task)
evaluate!(mach, resampling=Holdout(fraction_train=0.8), measure=[rms, rmslp1], verbosity=2)
# (MLJ.rms = 4.09, MLJ.rmslp1 = 0.26)

# Note that `evaluate!` has fitted `tuned_forest` on the training data
# (80% of all data), which means tuning the underlying `forest` (in
# this case using cross-validation) and then retraining `forest` on
# the full training data.

# We can view the optimal `forest` parameters:
fitted_params(mach)
fitted_params(mach).best_model
@more

# And plot the performance estimates for the grid search:
plot(mach)
heatmap(mach)


true
