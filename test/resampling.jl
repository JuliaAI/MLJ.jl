module TestResampling

# using Revise
using Test
using MLJ
using DataFrames

x1 = ones(4)
x2 = ones(4)
X = DataFrame(x1=x1, x2=x2)
y = [1.0, 1.0, 2.0, 2.0]

holdout = Holdout(fraction_train=0.5)
model = ConstantRegressor()
resampler = Resampler(tuning=holdout, model=model)
fitresult, cache, report = MLJ.fit(resampler, 1, X, y)
@test fitresult ≈ 1.0

holdout.fraction_train = 0.75
fitresult, cache, report = MLJ.update(resampler, 2, fitresult, cache, X, y)
@test fitresult ≈ 2/3

# resampler as trainable model:
resampler = Resampler(tuning=holdout, model=model)
trainable_resampler = trainable(resampler, X, y)
fit!(trainable_resampler, verbosity=2)
@test evaluate(trainable_resampler) ≈ 2/3



end
true
