module TestTuning

# using Revise
using Test
using MLJ
using MLJBase
import Random.seed!
seed!(1234)

@load KNNRegressor

include("foobarmodel.jl")

x1 = rand(100);
x2 = rand(100);
x3 = rand(100);
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(100);

@testset "2-parameter tune, no nesting" begin

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

    MLJBase.info_dict(tuned_model)

    tuned = machine(tuned_model, X, y)

    fit!(tuned)
    r = report(tuned)
    @test r.best_report isa NamedTuple{(:machines, :reports)}
    tuned_model.full_report=true
    fit!(tuned)
    report(tuned)
    fp = fitted_params(tuned)
    @test fp.best_fitted_params isa NamedTuple{(:machines, :fitted_params)}
    b = fp.best_model
    @test b isa MLJ.SimpleDeterministicCompositeModel

    measurements = tuned.report.measurements
    # should be all different:
    @test length(unique(measurements)) == length(measurements)

    @test length(b.transformer.features) == 3
    @test abs(b.model.lambda - 0.027825) < 1e-6

    # get the training error of the tuned_model:
    e = rms(y, predict(tuned, X))

    # check this error has same order of magnitude as best measurement
    # during tuning:
    r = e/tuned.report.best_measurement
    @test r < 10 && r > 0.1

    # test weights:
    tuned_model.weights = rand(length(y))
    fit!(tuned)
    @test tuned.report.measurements[1] != measurements[1]

end

#@testset "multiple resolutions" begin

    forest = EnsembleModel(atom=@load(KNNRegressor))
    pipe = @pipeline MyPipe(sel=FeatureSelector(), forest=forest)

    X = MLJ.table(rand(150, 2), names=[:x1, :x2])
    y = rand(150)

    r1 = range(pipe, :(sel.features), values=[[:x1,], [:x1, :x2]])
    r2 = range(pipe, :(forest.atom.K), lower = 1, upper=50)
    r3 = range(pipe, :(forest.bagging_fraction), lower=0.5, upper=1.0)
    ranges = [r1, r2, r3]

    holdout = Holdout(fraction_train=0.8)
    grid = Grid(resolution=5)

    tuned = TunedModel(model=pipe, tuning=grid,
                       resampling=holdout, measure=rms,
                       ranges=ranges)

    mach = machine(tuned, X, y)

    function num_grid_points(resolution)
        grid.resolution = resolution
        fit!(mach)
        return report(mach).measurements |> length
    end

    @test num_grid_points(6) == 72
    @test num_grid_points([:(forest.atom.K)=>7,
                           :(forest.bagging_fraction)=>4]) == 56
    @test num_grid_points([:(forest.atom.K)=>6, ]) == 60


    @testset "one parameter tune" begin
        ridge = FooBarRegressor()
        r = range(ridge, :lambda, lower=1e-7, upper=1e-3, scale=:log)
        tuned_model = TunedModel(model=ridge, ranges=r)
        tuned = machine(tuned_model, X, y)
        fit!(tuned)
        report(tuned)
    end

#end

@testset "nested parameter tune" begin
    tree_model = KNNRegressor()
    forest_model = EnsembleModel(atom=tree_model)
    r1 = range(forest_model, :(atom.K), lower=3, upper=4)
    r2 = range(forest_model, :bagging_fraction, lower=0.4, upper=1.0);
    self_tuning_forest_model = TunedModel(model=forest_model,
                                          tuning=Grid(resolution=3),
                                          resampling=Holdout(),
                                          ranges=[r1, r2],
                                          measure=rms)
    self_tuning_forest = machine(self_tuning_forest_model, X, y)
    fit!(self_tuning_forest, verbosity=2)
    r = report(self_tuning_forest)
    @test length(unique(r.measurements)) == 6
end

@testset "basic tuning with training weights" begin

    N = 100
    X = (x = rand(3N), );
    y = categorical(rand("abc", 3N));
    model = @load KNNClassifier
    r = range(model, :K, lower=2, upper=N)
    tuned_model = TunedModel(model=model,
                             measure=MLJ.BrierScore(),
                             resampling=Holdout(fraction_train=2/3),
                             range=r)

    # no weights:
    tuned = machine(tuned_model, X, y)
    fit!(tuned)
    best1 = fitted_params(tuned).best_model
    posterior1 = average([predict(tuned, X)...])

    # uniform weights:
    tuned = machine(tuned_model, X, y, fill(1, 3N))
    fit!(tuned)
    best2 = fitted_params(tuned).best_model
    posterior2 = average([predict(tuned, X)...])

    @test best1 == best2
    @test all([pdf(posterior1, c) ≈ pdf(posterior2, c) for c in levels(y)])

    # skewed weights:
    w = map(y) do η
        if η == 'a'
            return 2
        elseif η == 'b'
            return 4
        else
            return 1
        end
    end
    tuned = machine(tuned_model, X, y, w)
    fit!(tuned)
    best3 = fitted_params(tuned).best_model
    posterior3 = average([predict(tuned, X)...])

    # different tuning outcome:
    @test best1.K != best3.K

    # "posterior" is skewed appropriately in weighted case:
    @test abs(pdf(posterior3, 'b')/(2*pdf(posterior3, 'a'))  - 1) < 0.15
    @test abs(pdf(posterior3, 'b')/(4*pdf(posterior3, 'c'))  - 1) < 0.15


end


## LEARNING CURVE

@testset "learning curves" begin
    atom = FooBarRegressor()
    ensemble = EnsembleModel(atom=atom, n=50, rng=1)
    mach = machine(ensemble, X, y)
    r_lambda = range(ensemble, :(atom.lambda),
                     lower=0.0001, upper=0.1, scale=:log10)
    curve = MLJ.learning_curve!(mach; range=r_lambda)
    atom.lambda=0.3
    r_n = range(ensemble, :n, lower=10, upper=100)
    curve2 = MLJ.learning_curve!(mach; range=r_n)
    curve3 = learning_curve(ensemble, X, y; range=r_n)
    @test curve2.measurements ≈ curve3.measurements
end
end # module
true
