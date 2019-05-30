module TestResampling

# using Revise
using Test
using MLJ
using MLJBase
using DataFrames

x1 = ones(4)
x2 = ones(4)
X = DataFrame(x1=x1, x2=x2)
y = [1.0, 1.0, 2.0, 2.0]

@test CV(nfolds=6) == CV(nfolds=6)
@test CV(nfolds=5) != CV(nfolds=6)

# holdout:
@test MLJBase.show_as_constructed(Holdout)
holdout = Holdout(fraction_train=0.75)
model = ConstantRegressor()
resampler = Resampler(resampling=holdout, model=model, measure=rms)
fitresult, cache, report = MLJ.fit(resampler, 1, X, y)
@test fitresult ≈ 2/3

holdout = Holdout(shuffle=true)

mach = machine(model, X, y)
result = @test_logs((:info, r"^Evaluating using a holdout set"),
                    evaluate!(mach, resampling=holdout, measure=[rms, rmslp1]))
@test result isa NamedTuple
@test @test_logs((:info, r"^Evaluating using a holdout set"),
                 evaluate!(mach, resampling=holdout)) ≈ 2/3

x1 = ones(10)
x2 = ones(10)
X = DataFrame(x1=x1, x2=x2)
y = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]

# cv:
@test MLJBase.show_as_constructed(CV)
cv=CV(nfolds=5)
model = ConstantRegressor()
mach = machine(model, X, y)
result = @test_logs((:info, r"^Evaluating using cross-validation"),
                    evaluate!(mach, resampling=cv, measure=[rms, rmslp1]))
@test result isa NamedTuple
errs = @test_logs((:info, r"^Evaluating using cross-validation"),
                  evaluate!(mach, resampling=cv))
for e in errs
    @test e ≈ 1/2 || e ≈ 3/4
end
@test scitype_union(@test_logs((:info, r"^Evaluating using cross-validation"),
                               evaluate!(mach, resampling=CV(shuffle=true)))) == Continuous

## RESAMPLER AS MACHINE

# holdout:
X, y = datanow()
ridge_model = SimpleRidgeRegressor(lambda=20.0)
resampler = Resampler(resampling=holdout, model=ridge_model)
resampling_machine = machine(resampler, X, y)
@test_logs((:info, r"^Training"), fit!(resampling_machine))
e1=evaluate(resampling_machine)
mach = machine(ridge_model, X, y)
e1 ≈ @test_logs((:info, r"^Evaluating using a holdout set"),
                evaluate!(mach, holdout))
ridge_model.lambda=1.0
@test_logs((:info, r"^Updating"), (:info, r"^Evaluating using a holdout set"),
           fit!(resampling_machine, verbosity=2))
e2=evaluate(resampling_machine)
@test e1 != e2

# cv:
resampler = Resampler(resampling=cv, model=ridge_model)
resampling_machine = machine(resampler, X, y)
@test_logs (:info, r"^Training") fit!(resampling_machine)
e1=evaluate(resampling_machine)
mach = machine(ridge_model, X, y)
e1 ≈ @test_logs((:info, r"^Evaluating using cross-validation"),
                evaluate!(mach, cv))
ridge_model.lambda=10.0
@test_logs((:info, r"^Updating"), (:info, r"^Evaluating using cross-validation"),
           fit!(resampling_machine, verbosity=2))
e2=evaluate(resampling_machine)
@test mean(e1) != mean(e2)

@test MLJBase.package_name(Resampler) == "MLJ"
@test MLJBase.is_wrapper(Resampler)
rnd = randn(5)
@test evaluate(resampler, rnd) === rnd

end
true
