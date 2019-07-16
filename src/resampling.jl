## RESAMPLING STRATEGIES

abstract type ResamplingStrategy <: MLJType end

# resampling strategies are `==` if they have the same type and their
# field values are `==`:
function ==(s1::S, s2::S) where S <: ResamplingStrategy
    return all(getfield(s1, fld) == getfield(s2, fld) for fld in fieldnames(S))
end

"""
$TYPEDEF

Single train-test split with a (randomly selected) portion of the
data being selected for training and the rest for testing.

* `fraction_train` a number between 0 and 1 indicating the proportion
of the samples to use for training
* `shuffle` a boolean indicating whether to select the training samples
at random
* `rng` a random number generator to use
"""
@with_kw mutable struct Holdout <: ResamplingStrategy
    fraction_train::Float64 = 0.7
    shuffle::Bool = false
    rng::Union{Int,AbstractRNG} = Random.GLOBAL_RNG

    function Holdout(fraction_train, shuffle, rng)
        0 < fraction_train < 1 || error("`fraction_train` must be between 0 and 1.")
        return new(fraction_train, shuffle, rng)
    end
end

show_as_constructed(::Type{<:Holdout}) = true

"""
$TYPEDEF

Cross validation resampling where the data is (randomly) partitioned in `nfolds` folds
and the model is evaluated `nfolds` time, each time taking one fold for testing and the
other folds for training and the test  performances averaged.
For instance if `nfolds=3` then the data will be partitioned in three folds A, B and C
and the model will be trained three times, first with A and B and tested on C, then on
A, C and tested on B and finally on B, C and tested on A. The test performances are then
averaged over the three cases.
"""
@with_kw  mutable struct CV <: ResamplingStrategy
    nfolds::Int = 6
    parallel::Bool = true
    shuffle::Bool = false ## TODO: add seed/rng
    rng::Union{Int,AbstractRNG} = Random.GLOBAL_RNG
end

MLJBase.show_as_constructed(::Type{<:CV}) = true


## DIRECT EVALUATION METHODS

# We first define an `evaluate!` to directly generate estimates of
# performance according to some strategy
# `s::S<:ResamplingStrategy`. (For tuning we later define a
# `Resampler{S}` object that wraps a model in a resampling strategy
# and a measure.)  We need an `evaluate!` for each strategy.

"""
    evaluate!(mach, resampling=CV(), measure=nothing, operation=predict, force=false, verbosity=1)

Estimate the performance of a machine `mach` using the specified
`resampling` strategy (defaulting to 6-fold cross-validation) and `measure`,
which can be a single measure or vector.

Although evaluate! is mutating, `mach.model` and `mach.args` are
preserved.

Resampling and testing is based exclusively on data in `rows`, when
specified.

If no measure is specified, then `default_measure(mach.model)` is used, unless
this default is `nothing` and an error is thrown.

"""
evaluate!(mach::Machine;
          resampling=CV(), kwargs...) =
              evaluate!(mach, resampling; kwargs... )

# holdout:
function evaluate!(mach::Machine, resampling::Holdout;
                   measure=nothing, operation=predict,
                   rows=nothing, force=false,verbosity=1)

    if resampling.rng isa Integer
        rng = MersenneTwister(resampling.rng)
    else
        rng = resampling.rng
    end

    if measure === nothing
        _measures = default_measure(mach.model)
        if _measures === nothing
            error("You need to specify measure=... ")
        end
    else
        _measures = measure
    end

    X = mach.args[1]
    y = mach.args[2]
    length(mach.args) == 2 || error("Multivariate targets not yet supported.")

    unspecified_rows = (rows === nothing)
    all = unspecified_rows ? eachindex(y) : rows

    train, test = partition(all, resampling.fraction_train,
                            shuffle=resampling.shuffle, rng=rng)
    if verbosity > 0
        which_rows = ifelse(unspecified_rows, "Resampling from all rows. ",
                                              "Resampling from a subset of all rows. ")
        @info "Evaluating using a holdout set. \n" *
              "fraction_train=$(resampling.fraction_train) \n" *
              "shuffle=$(resampling.shuffle) \n" *
              "measure=$_measures \n" *
              "operation=$operation \n" *
              "$which_rows"
    end

    fit!(mach, rows=train, verbosity=verbosity-1, force=force)
    yhat = operation(mach, selectrows(X, test))

    if !(_measures isa AbstractVector)
        return _measures(yhat, y[test])
    end

    measure_values = [m(yhat, y[test]) for m in _measures]
    measure_names = Tuple(Symbol.(string.(_measures)))
    return NamedTuple{measure_names}(measure_values)

end

# cv:
function evaluate!(mach::Machine, resampling::CV;
                   measure=nothing, operation=predict,
                   rows=nothing, force=false, verbosity=1)

    if resampling.rng isa Integer
        rng = MersenneTwister(resampling.rng)
    else
        rng = resampling.rng
    end

    if measure === nothing
        _measures = default_measure(mach.model)
        if _measures === nothing
            error("You need to specify measure=... ")
        end
    else
        _measures = measure
    end

    X = mach.args[1]
    y = mach.args[2]
    length(mach.args) == 2 || error("Multivariate targets not yet supported.")

    unspecified_rows = (rows === nothing)
    all = unspecified_rows ? eachindex(y) : rows

    if verbosity > 0
        which_rows = ifelse(unspecified_rows, "Resampling from all rows. ",
                                              "Resampling from a subset of all rows. ")
        @info "Evaluating using cross-validation. \n" *
              "nfolds=$(resampling.nfolds). \n" *
              "shuffle=$(resampling.shuffle) \n" *
              "measure=$_measures \n" *
              "operation=$operation \n" *
              "$which_rows"
    end

    n_samples = length(all)
    nfolds = resampling.nfolds

    if resampling.shuffle
        shuffle!(rng, collect(all))
    end

    # number of samples per fold
    k = floor(Int, n_samples/nfolds)

    # function to return the measures for the fold `all[f:s]`:
    function get_measure(f, s)
        test = all[f:s] # TODO: replace with views?
        train = vcat(all[1:(f - 1)], all[(s + 1):end])
        fit!(mach; rows=train, verbosity=verbosity-1, force=force)
        yhat = operation(mach, selectrows(X, test))
        if !(_measures isa AbstractVector)
            return _measures(yhat, y[test])
        else
            return [m(yhat, y[test]) for m in _measures]
        end
    end

    firsts = 1:k:((nfolds - 1)*k + 1) # itr of first `test` rows index
    seconds = k:k:(nfolds*k)          # itr of last `test` rows  index

    if resampling.parallel && nworkers() > 1
        ## TODO: progress meter for distributed case
        if verbosity > 0
            @info "Distributing cross-validation computation " *
                  "among $(nworkers()) workers."
        end
        measure_values = @distributed vcat for n in 1:nfolds
            [get_measure(firsts[n], seconds[n])]
        end
    else
        if verbosity > 0
            p = Progress(nfolds + 1, dt=0, desc="Cross-validating: ",
                         barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
            next!(p)
            measure_values = [first((get_measure(firsts[n], seconds[n]), next!(p))) for n in 1:nfolds]
        else
            measure_values = [get_measure(firsts[n], seconds[n]) for n in 1:nfolds]
        end
    end

    if !(measure isa AbstractVector)
        return measure_values
    end

    # repackage measures:
    measures_reshaped = [[measure_values[i][j] for i in 1:nfolds] for j in 1:length(_measures)]

    measure_names = Tuple(Symbol.(string.(_measures)))
    return NamedTuple{measure_names}(Tuple(measures_reshaped))

end


## RESAMPLER - A MODEL WRAPPER WITH `evaluate` OPERATION

"""
$TYPEDEF

Resampler structure for the `TunedModel` `fit` defined in `tuning.jl`.
"""
mutable struct Resampler{S,M<:Supervised} <: Supervised
    model::M
    resampling::S # resampling strategy
    measure
    operation
end

MLJBase.package_name(::Type{<:Resampler}) = "MLJ"
MLJBase.is_wrapper(::Type{<:Resampler}) = true


Resampler(; model=ConstantRegressor(), resampling=Holdout(),
            measure=nothing, operation=predict) =
                Resampler(model, resampling, measure, operation)


function MLJBase.fit(resampler::Resampler, verbosity::Int, X, y)

    if resampler.measure === nothing
        measure = default_measure(resampler.model)
        if measure === nothing
            error("You need to specify measure=... ")
        end
    else
        measure = resampler.measure
    end

    mach = machine(resampler.model, X, y)

    fitresult = evaluate!(mach, resampler.resampling;
                         measure=measure,
                         operation=resampler.operation,
                         verbosity=verbosity-1)

    cache = (mach, deepcopy(resampler.resampling))
    report = NamedTuple()

    return fitresult, cache, report

end

# in special case of holdout, we can reuse the underlying model's
# machine, provided the training_fraction has not changed:
function MLJBase.update(resampler::Resampler{Holdout},
                        verbosity::Int, fitresult, cache, X, y)

    old_mach, old_resampling = cache

    if resampler.measure === nothing
        measure = default_measure(resampler.model)
        if measure === nothing
            error("You need to specify measure=... ")
        end
    else
        measure = resampler.measure
    end

    if old_resampling.fraction_train == resampler.resampling.fraction_train
        mach = old_mach
    else
        mach = machine(resampler.model, X, y)
        cache = (mach, deepcopy(resampler.resampling))
    end

    fitresult = evaluate!(mach, resampler.resampling;
                         measure=resampler.measure,
                         operation=resampler.operation,
                         verbosity=verbosity-1)

    report = NamedTuple

    return fitresult, cache, report

end

MLJBase.evaluate(model::Resampler, fitresult) = fitresult
