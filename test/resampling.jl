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
resampler = Resampler(tuning=holdout, model=model)
fitresult, cache, report = MLJ.fit(resampler, 1, X, y)
@test fitresult ≈ 2/3

# resampler as machine:
resampler = Resampler(tuning=holdout, model=model)
resampling_machine = machine(resampler, X, y)
fit!(resampling_machine, verbosity=2)
@test evaluate(resampling_machine) ≈ 2/3


end
true
