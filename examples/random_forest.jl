# Simple example using MLJ ensembling to construct a tuned random forest.
# Declare variables const for performance boost

# nproc()
using Plots
pyplot()

using MLJ
using DataFrames

task = load_boston()
Xraw, y = task()
X = DataFrame(Xraw)


## EVALUATE A SINGLE TREE

@load DecisionTreeRegressor
tree = DecisionTreeRegressor()
mach = machine(tree, X, y)
evaluate!(mach, resampling=Holdout(fraction_train=0.8), measure=rms)
# 7.06


## DEFINE THE RANDOM FOREST MODEL

forest = EnsembleModel(atom=tree, n=100)


## DEFINE A TUNING GRID

# inspect the nested hyperparameters of our composite model:
params(forest)

# define ranges for each parameter:
r1 = range(tree, :n_subfeatures, lower=1, upper=12)
r2 = range(forest, :bagging_fraction, lower=0.5, upper=1.0)

# define nested ranges, matching pattern of params(forest):
nested_ranges = (atom=(n_subfeatures=r1,), bagging_fraction=r2)


## WRAP THE MODEL IN A TUNING STRATEGY (a new model!)

tuning = Grid(resolution=12)
resampling = CV()
tuned_forest = TunedModel(model=forest, tuning=Grid(), resampling=CV(),
                          nested_ranges=nested_ranges, measure=rms)


## FIT THE WRAPPED FOREST MODEL TO TUNE IT
## AND TRAIN THE OPTIMAL MODEL ALL SUPPLIED DATA

train, test = partition(eachindex(y), 0.8) # 80:20 split

mach = machine(tuned_forest, X, y)
fit!(mach, rows=train)

plot(mach)
savefig("random_forest_tuning.png")
heatmap(mach)
savefig("random_forest_heatmap.png")

