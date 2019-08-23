module TestResampling

#  using Revise
using Test
using MLJ
using MLJBase
import Random.seed!
seed!(1234)

macro test_parallel(var, ex)
    quote
        $(esc(var)) = false
        $ex
        $(esc(var)) = true
        $ex
    end
end

@test CV(nfolds=6) == CV(nfolds=6)
@test CV(nfolds=5) != CV(nfolds=6)

@testset "checking measure/model compatibility" begin
    model = ConstantRegressor()
    y = rand(4)
    override=false
    @test MLJ._check_measure(:junk, :junk, :junk, :junk, true) == nothing
    @test_throws ArgumentError MLJ._check_measure(model, rms, y, predict, override)
    @test MLJ._check_measure(model, rms, y, predict_mean, override) == nothing
    @test MLJ._check_measure(model, rms, y, predict_median, override) == nothing
    y=categorical(collect("abc"))
    @test_throws(ArgumentError,
                 MLJ._check_measure(model, rms, y, predict_median, override))
    model = ConstantClassifier()
    @test_throws(ArgumentError,
                 MLJ._check_measure(model, misclassification_rate, y, predict, override))
    @test MLJ._check_measure(model, misclassification_rate, y,
                            predict_mode, override) == nothing
    model = MLJ.DeterministicConstantClassifier()
    @test_throws ArgumentError MLJ._check_measure(model, cross_entropy, y,
                            predict, override)
end

@testset "folds specified" begin
    @test_parallel dopar begin
    global dopar
    x1 = ones(10)
    x2 = ones(10)
    X = (x1=x1, x2=x2)
    y = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]

    my_rms(yhat, y) = sqrt(mean((yhat -y).^2))
    my_mav(yhat, y) = abs.(yhat - y)
    MLJ.reports_each_observation(::typeof(my_mav)) = true

    resampling = [(3:10, 1:2),
                  ([1, 2, 5, 6, 7, 8, 9, 10], 3:4),
                  ([1, 2, 3, 4, 7, 8, 9, 10], 5:6),
                  ([1, 2, 3, 4, 5, 6, 9, 10], 7:8),
                  (1:8, 9:10)]

    model = MLJ.DeterministicConstantRegressor()
    mach = machine(model, X, y)

    # check detection of incompatible measure (cross_entropy):
    @test_throws ArgumentError evaluate!(mach, resampling=resampling,
                                         measure=[cross_entropy, rmslp1], parallel=dopar)
    result = evaluate!(mach, resampling=resampling,
                       measure=[my_rms, my_mav, rmslp1], parallel=dopar)
    v = [1/2, 3/4, 1/2, 3/4, 1/2]
    @test result.per_fold[1] ≈ v
    @test result.per_fold[2] ≈ v
    @test result.per_fold[3][1] ≈ abs(log(2) - log(2.5))
    @test ismissing(result.per_observation[1])
    @test result.per_observation[2][1] ≈ [1/2, 1/2]
    @test result.per_observation[2][2] ≈ [3/4, 3/4]
    @test result.measurement[1] ≈ mean(v)
    @test result.measurement[2] ≈ mean(v)
    end
end

@testset "holdout" begin
    @test_parallel dopar begin
    global dopar
    x1 = ones(4)
    x2 = ones(4)
    X = (x1=x1, x2=x2)
    y = [1.0, 1.0, 2.0, 2.0]

    @test MLJBase.show_as_constructed(Holdout)
    holdout = Holdout(fraction_train=0.75)
    model = MLJ.DeterministicConstantRegressor()
    mach = machine(model, X, y)
    result = evaluate!(mach, resampling=holdout,
                       measure=[rms, rmslp1], parallel=dopar)
    result = evaluate!(mach, verbosity=0, resampling=holdout, parallel=dopar)
    result.measurement[1] ≈ 2/3

    # test direct evaluation of a model + data:
    result = evaluate(model, X, y, verbosity=0, resampling=holdout, measure=rms)
    @test result.measurement[1] ≈ 2/3

    X = (x=rand(100),)
    y = rand(100)
    mach = machine(model, X, y)
    evaluate!(mach, verbosity=0, Holdout(shuffle=true, rng=123), parallel=dopar)
    e1 = evaluate!(mach, verbosity=0, Holdout(shuffle=true), parallel=dopar).measurement[1]
    @test e1 != evaluate!(mach, verbosity=0, Holdout(), parallel=dopar).measurement[1]
    end
end

@testset "cv" begin
    @test_parallel dopar begin
    global dopar
    x1 = ones(10)
    x2 = ones(10)
    X = (x1=x1, x2=x2)
    y = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]

    @test MLJBase.show_as_constructed(CV)
    cv=CV(nfolds=5)
    model = MLJ.DeterministicConstantRegressor()
    mach = machine(model, X, y)
    result = evaluate!(mach, resampling=cv, measure=[rms, rmslp1], parallel=dopar)
    @test result.per_fold[1] ≈ [1/2, 3/4, 1/2, 3/4, 1/2]

    shuffled =  evaluate!(mach, resampling=CV(shuffle=true), parallel=dopar) # using rms default
    @test shuffled.measurement[1] != result.measurement[1]
    end
end

@testset "weights" begin
    @test_parallel dopar begin
    global dopar
    # cv:
    x1 = ones(4)
    x2 = ones(4)
    X = (x1=x1, x2=x2)
    y = [1.0, 2.0, 3.0, 1.0]
    w = 1:4
    cv=CV(nfolds=2)
    model = MLJ.DeterministicConstantRegressor()
    mach = machine(model, X, y)
    e = evaluate!(mach, resampling=cv, measure=l1,
                  weights=w, verbosity=0, parallel=dopar).measurement[1]
    @test e ≈ (1/3 + 13/14)/2
    end
end

@testset "resampler as machine" begin
    @test_parallel dopar begin
    global dopar
    N = 50
    X = (x1=rand(N), x2=rand(N), x3=rand(N))
    y = X.x1 -2X.x2 + 0.05*rand(N)

    ridge_model = FooBarRegressor(lambda=20.0)
    holdout = Holdout(fraction_train=0.75)
    resampler = Resampler(resampling=holdout, model=ridge_model, measure=mav)
    resampling_machine = machine(resampler, X, y)
    @test_logs((:info, r"^Training"), fit!(resampling_machine))
    e1=evaluate(resampling_machine).measurement[1]
    mach = machine(ridge_model, X, y)
    @test e1 ≈  evaluate!(mach, resampling=holdout,
                          measure=mav, verbosity=0, parallel=dopar).measurement[1]
    ridge_model.lambda=1.0
    fit!(resampling_machine, verbosity=2)
    e2=evaluate(resampling_machine).measurement[1]
    @test e1 != e2
    resampler.weights = rand(N)
    fit!(resampling_machine, verbosity=0)
    e3=evaluate(resampling_machine).measurement[1]
    @test e3 != e2

    @test MLJBase.package_name(Resampler) == "MLJ"
    @test MLJBase.is_wrapper(Resampler)
    rnd = randn(5)
    @test evaluate(resampler, rnd) === rnd
    end
end


end
true
