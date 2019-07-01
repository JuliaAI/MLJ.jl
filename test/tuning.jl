module TestTuning

# using Revise
using Test
using MLJ
# using UnicodePlots
import MLJBase

x1 = rand(100);
x2 = rand(100);
x3 = rand(100);
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(100);

sel = FeatureSelector()
stand = UnivariateStandardizer()
ridge = FooBarRegressor()
composite = MLJ.SimpleDeterministicCompositeModel(transformer=sel, model=ridge)

features_ = range(composite, :(transformer.features), values=[[:x1], [:x1, :x2], [:x2, :x3], [:x1, :x2, :x3]])
lambda_ = range(composite, :(model.lambda), lower=1e-6, upper=1e-1, scale=:log10)

ranges = [features_, lambda_]

holdout = Holdout(fraction_train=0.8)
grid = Grid(resolution=10)

tuned_model = TunedModel(model=composite, tuning=grid,
                         resampling=holdout, measure=rms,
                         ranges=ranges, full_report=false)

info(tuned_model)

tuned = machine(tuned_model, X, y)

fit!(tuned)
report(tuned)
tuned_model.full_report=true
fit!(tuned)
report(tuned)

b = fitted_params(tuned).best_model

measurements = tuned.report.measurements
# should be all different:
@test length(unique(measurements)) == length(measurements)

# the following should only fail in extremely rare event:
@test length(b.transformer.features) == 3

# get the training error of the tuned_model:
e = rms(y, predict(tuned, X))

# check this error has same order of magnitude as best measurement during tuning:
r = e/tuned.report.best_measurement
@test r < 10 && r > 0.1

# test single range:
ridge = FooBarRegressor()
r = range(ridge, :lambda, lower=1e-7, upper=1e-3, scale=:log)
tuned_model = TunedModel(model=ridge, ranges=r)
tuned = machine(tuned_model, X, y)
fit!(tuned)
report(tuned)

## LEARNING CURVE

X, y = datanow()
atom = FooBarRegressor()
ensemble = EnsembleModel(atom=atom, n=500)
mach = machine(ensemble, X, y)
r_lambda = range(ensemble, :(atom.lambda), lower=0.01, upper=0.3, scale=:log10)
curve = MLJ.learning_curve!(mach; range=r_lambda)

atom.lambda=0.3
r_n = range(ensemble, :n, lower=10, upper=100)
curve2 = MLJ.learning_curve!(mach; range=r_n)

end
true
