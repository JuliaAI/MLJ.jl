module TestTuning

# using Revise
using Test
using MLJ
using DataFrames

x1 = rand(100);
x2 = rand(100);
x3 = rand(100);
X = DataFrame(x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(100);

sel = FeatureSelector()
stand = UnivariateStandardizer()
ridge = RidgeRegressor()
composite = SimpleCompositeModel(transformer=sel, model=ridge)

features_ = strange(sel, :features,
                             values=[[:x1], [:x1, :x2], [:x2, :x3], [:x1, :x2, :x3]])
lambda_ = strange(ridge, :lambda,
                           lower=1e-6, upper=1e-1, scale=:log10)

nested_ranges = Params(:transformer => Params(features_), :model => Params(lambda_)) 

holdout = Holdout(fraction_train=0.8)
grid = Grid(resolution=10)

tuned_model = TunedModel(model=composite, tuning_strategy=grid, resampling_strategy=holdout,
                         nested_ranges=nested_ranges, report_measurements=true)

tuned = machine(tuned_model, X, y)

fit!(tuned)
b = tuned.report[:best_model]

measurements = tuned.report[:measurements]
# should be all different:
@test length(unique(measurements)) == length(measurements)

# the following should only fail in extremely rare event:
@test length(b.transformer.features) == 3

# get the training error of the tuned_model:
e = rms(y, predict(tuned, X))

# check this error has same order of magnitude as best measurement during tuning:
r = e/tuned.report[:best_measurement]
@test r < 10 && r > 0.1

ridge = RidgeRegressor()
tuned_model = TunedModel(model=ridge,
                          nested_ranges=Params(strange(ridge, :lambda, lower=0.01, upper=1.0)))
tuned = machine(tuned_model, X, y)
fit!(tuned)
tuned.report[:curve]

## LEARNING CURVE

atom = RidgeRegressor()
model = EnsembleModel(atom=atom)
r = range(atom, :lambda, lower=0.001, upper=1.0, scale=:log10)
nested_range = Params(:atom => Params(:lambda => r))
u, v = MLJ.learning_curve(model, X, y; nested_range = nested_range) 

end
true
