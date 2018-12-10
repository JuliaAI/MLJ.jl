module TestTuning

# using Revise
using Test
using MLJ
using DataFrames
import MultivariateStats

x1 = rand(100);
x2 = rand(100);
x3 = rand(100);
X = DataFrame(x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(100);

sel = FeatureSelector()
stand = UnivariateStandardizer()
ridge = RidgeRegressor()
composite = SimpleCompositeRegressor(transformer_X=sel, transformer_y=stand, regressor=ridge)

features_ = param_range(sel, :features, values = [[:x1], [:x1, :x2], [:x2, :x3], [:x1, :x2, :x3]])
lambda_ = param_range(ridge, :lambda, lower=1e-6, upper=1e-1, scale=:log10)

param_ranges = Params(:transformer_X => Params(:features => features_), :regressor => Params(:lambda => lambda_)) 

holdout = Holdout(fraction_train=0.8)
grid = Grid(resolution=3)

tuned_model = TunedModel(model=composite, tuning=grid, resampling=holdout,
                         param_ranges=param_ranges)

# tuned_model = TunedModel(model=ridge, tuning=grid, resampling=holdout,
#                          param_ranges=Params(:lambda=>param_range(ridge,:lambda,lower=0.05,upper=0.1)))

tuned_modelT = trainable(tuned_model, X, y)

fit!(tuned_modelT)
b = best(tuned_modelT)

measurements = tuned_modelT.report[:measurements]
# should be all different:
@test length(unique(measurements)) == length(measurements)

# best model will use all features with very high probability:
@test length(b.transformer_X.features) == 3

end
true
