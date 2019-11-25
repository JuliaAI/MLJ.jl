module TestResampling

# using Revise
using Test
using MLJ
using MLJBase
import MLJModels
import StatsBase
import Random.seed!
seed!(1234)

include("foobarmodel.jl")

@test CV(nfolds=6) == CV(nfolds=6)
@test CV(nfolds=5) != CV(nfolds=6)
@test MLJ.train_test_pairs(CV(), 1:10) !=
     MLJ.train_test_pairs(CV(shuffle=true), 1:10)
@test MLJ.train_test_pairs(Holdout(), 1:10) !=
     MLJ.train_test_pairs(Holdout(shuffle=true), 1:10)

@testset "checking measure/model compatibility" begin
    model = ConstantRegressor()
    y = rand(4)
    override=false
    @test MLJ._check_measure(:junk, :junk, :junk, :junk, true) == nothing
    @test_throws(ArgumentError,
                  MLJ._check_measure(model, rms, y, predict, override))
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
    model = MLJModels.DeterministicConstantClassifier()
    @test_throws ArgumentError MLJ._check_measure(model, cross_entropy, y,
                            predict, override)
end

@testset "folds specified" begin
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

    model = MLJModels.DeterministicConstantRegressor()
    mach = machine(model, X, y)

    # check detection of incompatible measure (cross_entropy):
    @test_throws ArgumentError evaluate!(mach, resampling=resampling,
                                         measure=[cross_entropy, rmslp1])
    result = evaluate!(mach, resampling=resampling,
                       measure=[my_rms, my_mav, rmslp1])
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

@testset "holdout" begin
    x1 = ones(4)
    x2 = ones(4)
    X = (x1=x1, x2=x2)
    y = [1.0, 1.0, 2.0, 2.0]

    @test MLJBase.show_as_constructed(Holdout)
    holdout = Holdout(fraction_train=0.75)
    model = MLJModels.DeterministicConstantRegressor()
    mach = machine(model, X, y)
    result = evaluate!(mach, resampling=holdout,
                       measure=[rms, rmslp1])
    result = evaluate!(mach, verbosity=0, resampling=holdout)
    result.measurement[1] ≈ 2/3

    # test direct evaluation of a model + data:
    result = evaluate(model, X, y, verbosity=0,
                      resampling=holdout, measure=rms)
    @test result.measurement[1] ≈ 2/3

    X = (x=rand(100),)
    y = rand(100)
    mach = machine(model, X, y)
    evaluate!(mach, verbosity=0,
              resampling=Holdout(shuffle=true, rng=123))
    e1 = evaluate!(mach, verbosity=0,
                   resampling=Holdout(shuffle=true)).measurement[1]
    @test e1 != evaluate!(mach, verbosity=0,
                          resampling=Holdout()).measurement[1]

end

@testset "cv" begin
    x1 = ones(10)
    x2 = ones(10)
    X = (x1=x1, x2=x2)
    y = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]

    @test MLJBase.show_as_constructed(CV)
    cv=CV(nfolds=5)
    model = MLJModels.DeterministicConstantRegressor()
    mach = machine(model, X, y)
    result = evaluate!(mach, resampling=cv, measure=[rms, rmslp1])
    @test result.per_fold[1] ≈ [1/2, 3/4, 1/2, 3/4, 1/2]

    shuffled =  evaluate!(mach, resampling=CV(shuffle=true)) # using rms default
    @test shuffled.measurement[1] != result.measurement[1]
end

@testset "stratified_cv" begin

    # check in explicit example:
    y = categorical(['c', 'a', 'b', 'a', 'c', 'x',
                 'c', 'a', 'a', 'b', 'b', 'b', 'b', 'b'])
    rows = [14, 13, 12, 11, 10, 9, 8, 7, 5, 4, 3, 2, 1]
    @test y[rows] == collect("bbbbbaaccabac")
    scv = StratifiedCV(nfolds=3)
    pairs = MLJ.train_test_pairs(scv, rows, nothing, y)
    @test pairs == [([12, 11, 10, 8, 5, 4, 3, 2, 1], [14, 13, 9, 7]),
                    ([14, 13, 10, 9, 7, 4, 3, 2, 1], [12, 11, 8, 5]),
                    ([14, 13, 12, 11, 9, 8, 7, 5], [10, 4, 3, 2, 1])]
    scv_random = StratifiedCV(nfolds=3, shuffle=true)
    pairs_random = MLJ.train_test_pairs(scv_random, rows, nothing, y)
    @test pairs != pairs_random

    # wrong target type throws error:
    @test_throws Exception MLJ.train_test_pairs(scv, rows, nothing, get.(y))

    # too many folds throws error:
    @test_throws Exception MLJ.train_test_pairs(StratifiedCV(nfolds=4),
                                                rows, nothing, y)

    # check class distribution is preserved in a larger randomized example:
    N = 30
    y = shuffle(vcat(fill(:a, N), fill(:b, 2N),
                        fill(:c, 3N), fill(:d, 4N))) |> categorical;
    d = fit(UnivariateFinite, y)
    pairs = MLJ.train_test_pairs(scv, 1:10N, nothing, y)
    folds = vcat(first.(pairs), last.(pairs))
    @test all([fit(UnivariateFinite, y[fold]) ≈ d for fold in folds])


end

@testset "sample weights in evaluation" begin

    # cv:
    x1 = ones(4)
    x2 = ones(4)
    X = (x1=x1, x2=x2)
    y = [1.0, 2.0, 3.0, 1.0]
    w = 1:4
    cv=CV(nfolds=2)
    model = MLJModels.DeterministicConstantRegressor()
    mach = machine(model, X, y)
    e = evaluate!(mach, resampling=cv, measure=l1,
                  weights=w, verbosity=0).measurement[1]
    @test e ≈ (1/3 + 13/14)/2

end

@testset "resampler as machine" begin
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
                          measure=mav, verbosity=0).measurement[1]
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

struct DummyResamplingStrategy <: MLJ.ResamplingStrategy end

@testset "custom strategy using resampling depending on X, y" begin
    function MLJ.train_test_pairs(resampling::DummyResamplingStrategy,
                              rows, X, y)
        train = filter(rows) do j
            y[j] == y[1]
        end
        test = setdiff(rows, train)
        return [(train, test),]
    end

    X = (x = rand(8), )
    y = categorical([:x, :y, :x, :x, :y, :x, :x, :y])
    @test MLJ.train_test_pairs(DummyResamplingStrategy(), 2:6, X, y) ==
        [([3, 4, 6], [2, 5]),]

    e = evaluate(ConstantClassifier(), X, y,
                 measure=misclassification_rate,
                 resampling=DummyResamplingStrategy(),
                 operation=predict_mode)
    @test e.measurement[1] ≈ 1.0
end

@testset "sample weights in training  and evaluation" begin
    yraw = ["Perry", "Antonia", "Perry", "Antonia", "Skater"]
    X = (x=rand(5),)
    y = categorical(yraw)
    w = [1, 10, 1, 10, 5]

    # without weights:
    mach = machine(ConstantClassifier(), X, y)
    e = evaluate!(mach, resampling=Holdout(fraction_train=0.6),
                  operation=predict_mode, measure=misclassification_rate)
    @test e.measurement[1] ≈ 1.0

    # with weights in training and evaluation:
    mach = machine(ConstantClassifier(), X, y, w)
    e = evaluate!(mach, resampling=Holdout(fraction_train=0.6),
              operation=predict_mode, measure=misclassification_rate)
    @test e.measurement[1] ≈ 1/3

    # with weights in training but overriden in evaluation:
    e = evaluate!(mach, resampling=Holdout(fraction_train=0.6),
              operation=predict_mode, measure=misclassification_rate,
                  weights = fill(1, 5))
    @test e.measurement[1] ≈ 1/2

    @test_throws(DimensionMismatch,
                 evaluate!(mach, resampling=Holdout(fraction_train=0.6),
                           operation=predict_mode,
                           measure=misclassification_rate,
                           weights = fill(1, 100)))

    @test_throws(ArgumentError,
                 evaluate!(mach, resampling=Holdout(fraction_train=0.6),
                           operation=predict_mode,
                           measure=misclassification_rate,
                           weights = fill('a', 5)))

    # resampling on a subset of all rows:
    model = @load KNNClassifier

    N = 200
    X = (x = rand(3N), );
    y = categorical(rand("abcd", 3N));
    w = rand(3N);
    rows = StatsBase.sample(1:3N, 2N, replace=false);
    Xsmall = selectrows(X, rows);
    ysmall = selectrows(y, rows);
    wsmall = selectrows(w, rows);

    mach1 = machine(model, Xsmall, ysmall, wsmall)
    e1 = evaluate!(mach1, resampling=CV(),
                   measure=misclassification_rate,
                   operation=predict_mode)

    mach2 = machine(model, X, y, w)
    e2 = evaluate!(mach2, resampling=CV(),
                   measure=misclassification_rate,
                   operation=predict_mode,
                   rows=rows)

    @test e1.per_fold ≈ e2.per_fold

    # resampler as machine with evaluation weights not specified:
    resampler = Resampler(model=model, resampling=CV();
                          measure=misclassification_rate,
                          operation=predict_mode)
    resampling_machine = machine(resampler, X, y, w)
    fit!(resampling_machine)
    e1 = evaluate(resampling_machine).measurement[1]
    mach = machine(model, X, y, w)
    e2 = evaluate!(mach, resampling=CV();
                   measure=misclassification_rate,
                   operation=predict_mode).measurement[1]
    @test e1 ≈ e2

    # resampler as machine with evaluation weights specified:
    weval = rand(3N);
    resampler = Resampler(model=model, resampling=CV();
                          measure=misclassification_rate,
                          operation=predict_mode,
                          weights=weval)
    resampling_machine = machine(resampler, X, y, w)
    fit!(resampling_machine)
    e1 = evaluate(resampling_machine).measurement[1]
    mach = machine(model, X, y, w)
    e2 = evaluate!(mach, resampling=CV();
                   measure=misclassification_rate,
                   operation=predict_mode,
                   weights=weval).measurement[1]
    @test e1 ≈ e2

end



end
true
