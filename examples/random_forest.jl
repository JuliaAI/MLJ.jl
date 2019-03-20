# Simple example using MLJ ensembling to construct a tuned random forest.
# Declare variables const for performance boost

# uncomment 2 lines for parallelized ensemble building:
# using Distributed
# addprocs()

# using Revise
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

forest = EnsembleModel(atom=tree)


## GET IDEA OF NUMBER OF TREES NEEDED

# reduce number of features sampled at each node to a common default
# value used in ensembles:
tree.n_subfeatures = round(Int, sqrt(size(X, 2)))

# construct machine:
mach = machine(forest, X, y)

# range for forest.n:
r = range(forest, :n, lower=10, upper=500)

# generate learning curve, rms loss vs n:
curve = learning_curve!(mach, nested_range=(n=r,), measure=rms)

using UnicodePlots
lineplot(curve.parameter_values, curve.measurements)

   #     ┌────────────────────────────────────────┐ 
   # 7.8 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   #     │⢰⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   #     │⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   #     │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   #     │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   #     │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   #     │⠀⡇⠀⠀⠀⢀⠔⢲⠀⠀⡠⠃⠱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   #     │⠀⢱⠀⠀⠀⡜⠀⠀⢣⠜⠀⠀⠀⠘⠔⠊⠑⠒⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   #     │⠀⢸⠀⠀⢰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   #     │⠀⢸⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢢⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   #     │⠀⢸⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡀⡠⡀⠀⡠⠊⠑⠢⢄⠔⠉⠉⠉⠉⠢⢄⣀⠀│ 
   #     │⠀⠈⣦⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠋⠀⠘⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉│ 
   #     │⠀⠀⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   #     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   # 7.3 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   #     └────────────────────────────────────────┘ 
   #     0                                      500

forest.n = 300

                     
## DEFINE A TUNING GRID

# inspect the nested hyperparameters of our composite model:
params(forest)

# define ranges for each parameter:
r1 = range(tree, :n_subfeatures, lower=1, upper=12)
r2 = range(forest, :bagging_fraction, lower=0.4, upper=1.0)

# define nested ranges, matching pattern of params(forest):
nested_ranges = (atom=(n_subfeatures=r1,), bagging_fraction=r2)


## WRAP THE MODEL IN A TUNING STRATEGY (creates a new model!)

tuning = Grid(resolution=12)
resampling = CV()
tuned_forest = TunedModel(model=forest, tuning=tuning, resampling=resampling,
                          nested_ranges=nested_ranges, measure=rms)


## FIT THE WRAPPED FOREST MODEL TO TUNE IT
## AND TRAIN THE OPTIMAL MODEL ON ALL SUPPLIED DATA

train, test = partition(eachindex(y), 0.8) # 80:20 split

mach = machine(tuned_forest, X, y)
fit!(mach, rows=train) # tuned and trained on train only

# view optimal model parameters:
show(fitted_params(mach).best_model, 2)

# uncomment 6 lines for plotting the tuning results:
# using Plots
# pyplot()
# plot(mach)
# savefig("random_forest_tuning.png")
# heatmap(mach)
# savefig("random_forest_heatmap.png")


## EVALUATE THE RANDOM FOREST 

yhat = predict(mach, X[test,:]);
rms(yhat, y[test])
# 4.01

true
