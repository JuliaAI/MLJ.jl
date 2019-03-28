# two-parameter tuning plot (with nested parameters)

using MLJ
using DataFrames, Statistics

Xraw = rand(300,3)
y = exp(Xraw[:,1] - Xraw[:,2] - 2Xraw[:,3] + 0.1*rand(300))
X = DataFrame(Xraw)

train, test = partition(eachindex(y), 0.70); # 70:30 split

knn_model = KNNRegressor(K=10)
knn = machine(knn_model, X, y)

ensemble_model = EnsembleModel(atom=knn_model, n=20)
ensemble = machine(ensemble_model, X, y)

B_range = range(ensemble_model, :bagging_fraction, lower= 0.5, upper=1.0, scale = :linear)
K_range = range(knn_model, :K, lower=1, upper=100, scale=:log10)
nested_ranges = (atom = (K = K_range,), bagging_fraction = B_range)

tuning = Grid(resolution=12)
resampling = Holdout(fraction_train=0.8)

tuned_ensemble_model = TunedModel(model=ensemble_model, 
    tuning=tuning, resampling=resampling, nested_ranges=nested_ranges)

tuned_ensemble = machine(tuned_ensemble_model, X[train,:], y[train])
fit!(tuned_ensemble);

# data needed for plotting is here:
tuned_ensemble.report

# uncomment 4 lines to see plots
# using Plots
# pyplot()
# plot(tuned_ensemble)
# heatmap(tuned_ensemble)

true

