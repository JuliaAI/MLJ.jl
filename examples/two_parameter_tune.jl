# To show how to accesss the report of a TunedModel machine for
# plotting purposed, in the case a of a two-parameter tune.

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

r = report(tuned_ensemble)
keys(r)
xlab, ylab = r.parameter_names
xscale, yscale = r.parameter_scales
x = r.parameter_values[:,1]
y = r.parameter_values[:,2]
z = r.measurements

@assert length(x) == length(y) && length(y) == length(z)

# possible scale values:
?MLJ.scale



