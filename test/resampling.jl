module TestResampling

# using Revise
using Test
using MLJ
using DataFrames

x1 = ones(4)
x2 = ones(4)
X = DataFrame(x1=x1, x2=x2)
y = [1.0, 1.0, 2.0, 2.0]

# holdout:
holdout = Holdout(fraction_train=0.75)
model = ConstantRegressor()
resampler = Resampler(resampling=holdout, model=model, measures=rms)
fitresult, cache, report = MLJ.fit(resampler, 1, X, y)
@test fitresult ≈ 2/3

holdout = Holdout(shuffle=true)

mach = machine(model, X, y)
result = evaluate!(mach, resampling=holdout, measures=[rms, rmslp1])
@test result isa NamedTuple
@test evaluate!(mach, resampling=holdout) ≈ 2/3

x1 = ones(10)
x2 = ones(10)
X = DataFrame(x1=x1, x2=x2)
y = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]

# cv:
cv=CV(nfolds=5)
model = ConstantRegressor()
mach = machine(model, X, y)
result = evaluate!(mach, resampling=cv, measures=[rms, rmslp1])
@test result isa NamedTuple
errs = evaluate!(mach, resampling=cv)
for e in errs
    @test e ≈ 1/2 || e ≈ 3/4
end
         
## RESAMPLER AS MACHINE

# holdout:
# X, y = datanow()
ridge_model = RidgeRegressor(lambda=20.0)
resampler = Resampler(resampling=holdout, model=ridge_model)
resampling_machine = machine(resampler, X, y)
fit!(resampling_machine)
e1=evaluate(resampling_machine)
mach = machine(ridge_model, X, y)
e1 ≈ evaluate!(mach, holdout)
ridge_model.lambda=1.0
fit!(resampling_machine, verbosity=2)
e2=evaluate(resampling_machine)
@test e1 != e2

# cv:
resampler = Resampler(resampling=cv, model=ridge_model)
resampling_machine = machine(resampler, X, y)
fit!(resampling_machine)
e1=evaluate(resampling_machine)
mach = machine(ridge_model, X, y)
e1 ≈ evaluate!(mach, cv)
ridge_model.lambda=10.0
fit!(resampling_machine, verbosity=2)
e2=evaluate(resampling_machine)
@test mean(e1) != mean(e2)

end
true
