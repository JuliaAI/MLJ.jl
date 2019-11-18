## RESAMPLING STRATEGIES

abstract type ResamplingStrategy <: MLJType end
show_as_constructed(::Type{<:ResamplingStrategy}) = true

# resampling strategies are `==` if they have the same type and their
# field values are `==`:
function ==(s1::S, s2::S) where S <: ResamplingStrategy
    return all(getfield(s1, fld) == getfield(s2, fld) for fld in fieldnames(S))
end

# fallbacks:
train_test_pairs(s::ResamplingStrategy, rows, X, y, w) =
    train_test_pairs(s, rows, X, y)
train_test_pairs(s::ResamplingStrategy, rows, X, y) =
    train_test_pairs(s, rows, y)
train_test_pairs(s::ResamplingStrategy, rows, y) =
    train_test_pairs(s, rows)

"""
    holdout = Holdout(; fraction_train=0.7,
                         shuffle=false,
                         rng=Random.GLOBAL_RNG)

Holdout resampling strategy, for use in `evaluate!`, `evaluate` and in tuning.

    train_test_pairs(holdout, rows)

Returns the pair `[(train, test)]`, where `train` and `test` are
vectors such that `rows=vcat(train, test)` and
`length(train)/length(test) ≈ fraction_train`.

If `rng` is an integer, then `MersenneTwister(rng)` is the random
number generator used for shuffling rows. Otherwise some `AbstractRNG`
object is expected.

"""
struct Holdout <: ResamplingStrategy
    fraction_train::Float64
    shuffle::Bool
    rng::Union{Int,AbstractRNG}

    function Holdout(fraction_train, shuffle, rng)
        0 < fraction_train < 1 ||
            error("`fraction_train` must be between 0 and 1.")
        return new(fraction_train, shuffle, rng)
    end
end

# Keyword Constructor
function Holdout(; fraction_train::Float64=0.7,
                   shuffle::Bool=false,
                   rng::Union{Int,AbstractRNG}=Random.GLOBAL_RNG)
    Holdout(fraction_train, shuffle, rng)
end


function train_test_pairs(holdout::Holdout, rows)
    if holdout.rng isa Integer
        rng = MersenneTwister(holdout.rng)
    else
        rng = holdout.rng
    end
    train, test = partition(rows, holdout.fraction_train,
                             shuffle=holdout.shuffle, rng=rng)
    return [(train, test),]
end


"""
    cv = CV(; nfolds=6,  shuffle=false, rng=Random.GLOBAL_RNG)

Cross-validation resampling strategy, for use in `evaluate!`,
`evaluate` and tuning.

    train_test_pairs(cv, rows)

Returns an `nfolds`-length iterator of `(train, test)` pairs of
vectors (row indices), where each `train` and `test` is a sub-vector
of `rows`. The `test` vectors are mutually exclusive and exhaust
`rows`. Each `train` vector is the complement of the
corresponding `test` vector. With no shuffling, the order of `rows` is
preserved, in the sense that `rows` coincides precisely with the
concatenation of the `test` vectors, in the order they are
generated. All but the last `test` vector have equal length.

Declaring `shuffle=true` results in `rows` being shuffled first. If
`rng` is an integer, then `MersenneTwister(rng)` is the random number
generator used for shuffling `rows`. Otherwise some `AbstractRNG`
object is expected.

"""
struct CV <: ResamplingStrategy
    nfolds::Int
    shuffle::Bool
    rng::Union{Int,AbstractRNG}
    function CV(nfolds, shuffle, rng)
        nfolds > 1 || error("Must have nfolds > 1. ")
        return new(nfolds, shuffle, rng)
    end
end

# Constructor with keywords
CV(; nfolds::Int=6,  shuffle::Bool=false,
   rng::Union{Int,AbstractRNG}=Random.GLOBAL_RNG) =
       CV(nfolds, shuffle, rng)

function train_test_pairs(cv::CV, rows)
    if cv.rng isa Integer
        rng = MersenneTwister(cv.rng)
    else
        rng = cv.rng
    end

    n_observations = length(rows)
    nfolds = cv.nfolds

    if cv.shuffle
        rows=shuffle!(rng, collect(rows))
    end

    # number of observations per fold
    k = floor(Int, n_observations/nfolds)
    k > 0 || error("Inusufficient data for $nfolds-fold cross-validation.\n"*
                   "Try reducing nfolds. ")

    # define the (trainrows, testrows) pairs:
    firsts = 1:k:((nfolds - 1)*k + 1) # itr of first `test` rows index
    seconds = k:k:(nfolds*k)          # itr of last  `test` rows index

    ret = map(1:nfolds) do k
        f = firsts[k]
        s = seconds[k]
        k < nfolds || (s = n_observations)
        return (vcat(rows[1:(f - 1)], rows[(s + 1):end]), # trainrows
                rows[f:s])                                # testrows
    end

    return ret
end

"""
    stratified_cv = StratifiedCV(; nfolds=6,  
                                   shuffle=false, 
                                   rng=Random.GLOBAL_RNG)

Stratified cross-validation resampling strategy, for use in
`evaluate!`, `evaluate` and in tuning. Applies only to classification
problems (`OrderedFactor` or `Multiclass` targets).

    train_test_pairs(stratified_cv, rows, y)
 
Returns an `nfolds`-length iterator of `(train, test)` pairs of
vectors (row indices) where each `train` and `test` is a sub-vector of
`rows`. The `test` vectors are mutually exclusive and exhaust
`rows`. Each `train` vector is the complement of the corresponding
`test` vector. 

Unlike regular cross-validation, the distribution of the levels of the
target `y` corresponding to each `train` and `test` is constrained, as
far as possible, to replicate that of `y[rows]` as a whole.

Specifically, the data is split into a number of groups on which `y`
is constant, and each individual group is resampled according to the
ordinary cross-validation strategy `CV(nfolds=nfolds)`. To obtain the
final `(train, test)` pairs of row indices, the per-group pairs are
collated in such a way that each collated `train` and `test` respects
the original order of `rows` (after shuffling, if `shuffle=true`).

If `rng` is an integer, then `MersenneTwister(rng)` is the random
number generator used for shuffling rows. Otherwise some `AbstractRNG`
object is expected.

"""
struct StratifiedCV <: ResamplingStrategy
    nfolds::Int
    shuffle::Bool
    rng::Union{Int,AbstractRNG}
    function StratifiedCV(nfolds, shuffle, rng)
        nfolds > 1 || error("Must have nfolds > 1. ")
        return new(nfolds, shuffle, rng)
    end
end

# Constructor with keywords
StratifiedCV(; nfolds::Int=6,  shuffle::Bool=false,
   rng::Union{Int,AbstractRNG}=Random.GLOBAL_RNG) =
       StratifiedCV(nfolds, shuffle, rng)


function train_test_pairs(stratified_cv::StratifiedCV, rows, X, y)
    if stratified_cv.rng isa Integer
        rng = MersenneTwister(stratified_cv.rng)
    else
        rng = stratified_cv.rng
    end

    n_observations = length(rows)
    nfolds = stratified_cv.nfolds

    if stratified_cv.shuffle
        rows=shuffle!(rng, collect(rows))
    end

    st = scitype(y)
    st <: AbstractArray{<:Finite} ||
        error("Supplied target has scitpye $st but stratified "*
              "cross-validation applies only to classification problems. ")


    freq_given_level = countmap(y[rows])
    minimum(values(freq_given_level)) >= nfolds ||
        error("The number of observations for which the target takes on a "*
              "given class must, for each class, exceed `nfolds`. Try "*
              "reducing `nfolds`. ")
    
    levels_seen = keys(freq_given_level) |> collect

    cv = CV(nfolds=nfolds)

    # the target is constant on each stratum, a subset of `rows`:
    class_rows = [rows[y[rows] .== c] for c in levels_seen]

    # get the cv train/test pairs for each level:
    train_test_pairs_per_level = (MLJ.train_test_pairs(cv, class_rows[m])
                              for m in eachindex(levels_seen))

    # just the train rows in each level:
    trains_per_level = map(x -> first.(x),
                           train_test_pairs_per_level)

    # just the test rows in each level:
    tests_per_level  = map(x -> last.(x),
                                train_test_pairs_per_level)
    
    # for each fold, concatenate the train rows over levels: 
    trains_per_fold = map(x->vcat(x...), zip(trains_per_level...))
    
    # for each fold, concatenate the test rows over levels: 
    tests_per_fold = map(x->vcat(x...), zip(tests_per_level...))
    
    # restore ordering specified by rows:
    trains_per_fold = map(trains_per_fold) do train
        filter(in(train), rows)
    end
    tests_per_fold = map(tests_per_fold) do test
        filter(in(test), rows)
    end
    
    # re-assemble:
    return zip(trains_per_fold, tests_per_fold) |> collect

end


## DIRECT EVALUATION METHODS

"""
    evaluate!(mach,
              resampling=CV(),
              measure=nothing,
              weights=nothing,
              operation=predict,
              acceleration=DEFAULT_RESOURCE[],
              force=false,
              verbosity=1)

Estimate the performance of a machine `mach` wrapping a supervised
model in data, using the specified `resampling` strategy (defaulting
to 6-fold cross-validation) and `measure`, which can be a single
measure or vector.

Do `subtypes(MLJ.ResamplingStrategy)` to obtain a list of available resampling
strategies. If `resampling` is not an object of type
`MLJ.ResamplingStrategy`, then a vector of pairs (of the form
`(train_rows, test_rows)` is expected. For example, setting

    resampling = [(1:100), (101:200)),
                   (101:200), (1:100)]

gives two-fold cross-validation using the first 200 rows of data.

If `resampling isa MLJ.ResamplingStrategy` then one may optionally
restrict the data used in evaluation by specifying `rows`.

An optional `weights` vector may be passed for measures that support
sample weights (`MLJ.supports_weights(measure) == true`), which
is ignored by those that don't.

User-defined measures are supported; see the manual for details.

If no measure is specified, then `default_measure(mach.model)` is
used, unless this default is `nothing` and an error is thrown.

The `acceleration` keyword argument is used to specify the compute resource (a
subtype of `ComputationalResources.AbstractResource`) that will be used to
accelerate/parallelize the resampling operation.

Although evaluate! is mutating, `mach.model` and `mach.args` are
untouched.

"""
evaluate!(mach::Machine{<:Supervised};
          resampling=CV(), kwargs...) =
              evaluate!(mach, resampling; kwargs... )

"""
    evaluate(model, X, y; measure=nothing, options...)

Evaluate the performance of a supervised model `model` on input data
`X` and target `y`. See the machine version `evaluate!` for options.

"""
evaluate(model::Supervised, args...; kwargs...) =
    evaluate!(machine(model, args...); kwargs...)

const AbstractRow = Union{AbstractVector{<:Integer}, Colon}
const TrainTestPair = Tuple{AbstractRow,AbstractRow}
const TrainTestPairs = AbstractVector{<:TrainTestPair}

function _check_measure(model, measure, y, operation, override)

    override && (return nothing)

    T = scitype(y)

    T == Unknown && (return nothing)
    target_scitype(measure) == Unknown && (return nothing)
    prediction_type(measure) == :unknown && (return nothing)

    avoid = "\nTo override measure checks, set check_measure=false. "

    T <: target_scitype(measure) ||
        throw(ArgumentError(
            "\nscitype of target = $T but target_scitype($measure) = "*
            "$(target_scitype(measure))."*avoid))

    if model isa Probabilistic
        if operation == predict
            if prediction_type(measure) != :probabilistic
                suggestion = ""
                if target_scitype(measure) <: Finite
                    suggestion = "\nPerhaps you want to set operation="*
                    "predict_mode. "
                elseif target_scitype(measure) <: Continuous
                    suggestion = "\nPerhaps you want to set operation="*
                    "predict_mean or operation=predict_median. "
                else
                    suggestion = ""
                end
                throw(ArgumentError(
                    "\n$model <: Probabilistic but prediction_type($measure) = "*
                      ":$(prediction_type(measure)). "*suggestion*avoid))
            end
        end
    end

    model isa Deterministic && prediction_type(measure) != :deterministic &&
        throw(ArgumentError("$model <: Deterministic but prediction_type($measure) ="*
              ":$(prediction_type(measure))."*avoid))

    return nothing

end

function _evaluate!(func::Function, res::CPU1, nfolds, verbosity)
    p = Progress(nfolds + 1, dt=0, desc="Evaluating over $nfolds folds: ",
                 barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
    verbosity > 0 && next!(p)
    return reduce(vcat, (func(k, p, verbosity) for k in 1:nfolds))
end
function _evaluate!(func::Function, res::CPUProcesses, nfolds, verbosity)
    # TODO: use pmap here ?:
    return @distributed vcat for k in 1:nfolds
        func(k)
    end
end
@static if VERSION >= v"1.3.0-DEV.573"
    function _evaluate!(func::Function, res::CPUThreads, nfolds, verbosity)
        task_vec = [Threads.@spawn func(k) for k in 1:nfolds]
        return fetch.(task_vec)
    end
end

# if `resampling` is not a ResamplingStrategy object:
function evaluate!(mach::Machine, resampling;
                   measure=nothing, weights=nothing,
                   operation=predict, acceleration=DEFAULT_RESOURCE[],
                   rows=nothing, force=false,
                   check_measure=true, verbosity=1)

    resampling isa TrainTestPairs ||
        error("`resampling` must be an "*
              "MLJ.ResamplingStrategy or tuple of pairs "*
              "of the form `(train_rows, test_rows)`")

    rows === nothing ||
        error("You cannot specify `rows` unless `resampling "*
              "isa MLJ.ResamplingStrategy`. ")

    if measure === nothing
        candidate = default_measure(mach.model)
        candidate ===  nothing && error("You need to specify measure=... ")
        measures = [candidate, ]
    elseif !(measure isa AbstractVector)
        measures = [measure, ]
    else
        measures = measure
    end

    X = mach.args[1]
    y = mach.args[2]

    [_check_measure(mach.model, m, y, operation, !check_measure) for m in measures]

    if verbosity >= 0 && weights !== nothing
        unsupported = filter(measures) do m
            !supports_weights(m)
        end
        if !isempty(unsupported)
            unsupported_as_string = string(unsupported[1])
            unsupported_as_string *=
                reduce(*, [string(", ", m) for m in unsupported[2:end]])
            @warn "weights ignored in evaluations of the following measures, "*
            "as unsupported: \n$unsupported_as_string "
        end
    end

    nfolds = length(resampling)
    nmeasures = length(measures)

    function get_measurements(k)
        train, test = resampling[k]
        fit!(mach; rows=train, verbosity=verbosity-1, force=force)
        Xtest = selectrows(X, test)
        ytest = selectrows(y, test)
        if weights == nothing
            wtest = nothing
        else
            wtest = weights[test]
        end
        yhat = operation(mach, Xtest)
        return [value(m, yhat, Xtest, ytest, wtest)
                for m in measures]
    end
    function get_measurements(k, p, verbosity) # p = progress meter
        ret = get_measurements(k)
        verbosity > 0 && next!(p)
        return ret
    end

    measurements_flat = if acceleration isa CPUProcesses
        ## TODO: progress meter for distributed case
        if verbosity > 0
            @info "Distributing cross-validation computation " *
                  "among $(nworkers()) workers."
        end
    end
    measurements_flat = _evaluate!(get_measurements, acceleration, nfolds, verbosity)

    # in the following rows=folds, columns=measures:
    measurements_matrix = permutedims(
        reshape(measurements_flat, (nmeasures, nfolds)))

    # measurements for each observation:
    per_observation = map(1:nmeasures) do k
        m = measures[k]
        if reports_each_observation(m)
            [measurements_matrix[:,k]...]
        else
            missing
        end
    end

    # measurements for each fold:
    per_fold = map(1:nmeasures) do k
        m = measures[k]
        if reports_each_observation(m)
            broadcast(MLJBase.aggregate, per_observation[k], [m,])
        else
            [measurements_matrix[:,k]...]
        end
    end

    # overall aggregates:
    per_measure = map(1:nmeasures) do k
        m = measures[k]
        MLJBase.aggregate(per_fold[k], m)
    end

    ret = (measure=measures,
           measurement=per_measure,
           per_fold=per_fold,
           per_observation=per_observation)

    verbosity < 1 || nmeasures < 2 ||
        pretty(selectcols(ret, 1:2), showtypes=false)

    return ret

end

function actual_rows_and_weights(rows, weights, N, verbosity)
    unspecified_rows = (rows === nothing)
    _rows = unspecified_rows ? (1:N) : rows
    if weights == nothing
        _weights = nothing
    else
        _weights = weights[_rows]
    end
    unspecified_rows || @info "Creating subsamples from a subset of all rows. "
    return _rows, _weights
end

# evaluation when ResamplingStrategy is passed (instead of train/test rows):
function evaluate!(mach::Machine, resampling::ResamplingStrategy;
                   weights=nothing, rows=nothing, verbosity=1, kwargs...)

    y = mach.args[2]
    _rows, _weights =
        actual_rows_and_weights(rows, weights, length(y), verbosity)

    return evaluate!(mach::Machine,
               train_test_pairs(resampling, _rows, mach.args...);
               weights=_weights, verbosity=verbosity, kwargs...)

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
    weights::Union{Nothing,AbstractVector{<:Real}}
    operation
    acceleration::AbstractResource
end

MLJBase.package_name(::Type{<:Resampler}) = "MLJ"
MLJBase.is_wrapper(::Type{<:Resampler}) = true


Resampler(; model=ConstantRegressor(), resampling=Holdout(),
            measure=nothing, weights=nothing, operation=predict,
            acceleration=DEFAULT_RESOURCE[]) =
                Resampler(model, resampling, measure, weights, operation,
                          acceleration)


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
                          measure=measure, weights=resampler.weights,
                          operation=resampler.operation,
                          verbosity=verbosity-1,
                          acceleration=resampler.acceleration)

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
                          weights=resampler.weights,
                          operation=resampler.operation,
                          verbosity=verbosity-1)

    report = NamedTuple

    return fitresult, cache, report

end

MLJBase.input_scitype(::Type{<:Resampler{S,M}}) where {S,M} = MLJBase.input_scitype(M)
MLJBase.target_scitype(::Type{<:Resampler{S,M}}) where {S,M} = MLJBase.target_scitype(M)


## EVALUATE FOR MODEL + DATA

MLJBase.evaluate(model::Resampler, fitresult) = fitresult
