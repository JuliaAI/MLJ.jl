module TestResampling

# using Revise
using Test
using MLJ
using DataFrames

x1 = ones(4)
x2 = ones(4)
X = DataFrame(x1=x1, x2=x2)
y = [1.0, 1.0, 2.0, 2.0]

holdout = Holdout(fraction_train=0.75)
model = ConstantRegressor()
resampler = Resampler(resampling_strategy=holdout, model=model)
fitresult, cache, report = MLJ.fit(resampler, 1, X, y)
@test fitresult ≈ 2/3

mach = machine(model, X, y)
@test evaluate(mach, holdout) ≈ 2/3


# resampler as machine:
X, y = datanow()
ridge_model = RidgeRegressor(lambda=20.0)
resampler = Resampler(resampling_strategy=holdout, model=ridge_model)
resampling_machine = machine(resampler, X, y)
fit!(resampling_machine, verbosity=2)
e1=evaluate(resampling_machine)
mach = machine(ridge_model, X, y)
e1 ≈ evaluate(mach, holdout)
ridge_model.lambda=1.0
fit!(resampling_machine, verbosity=2)
e2=evaluate(resampling_machine)
@test e1 != e2

mean(evaluate(mach, CV()))

end
true
