## RESAMPLING STRATEGIES

abstract type ResamplingStrategy <: MLJType end

# resampling strategies are `==` if they have the same type and their
# field values are `==`:
function ==(s1::S, s2::S) where S<:ResamplingStrategy
    ret = true
    for fld in fieldnames(S)
        ret = ret && getfield(s1, fld) == getfield(s2, fld)
    end
    return ret
end

mutable struct Holdout <: ResamplingStrategy
    fraction_train::Float64
    shuffle::Bool
    function Holdout(fraction_train, shuffle)
        0 < fraction_train && fraction_train < 1 ||
            error("fraction_train must be between 0 and 1.")
        return new(fraction_train, shuffle)
    end
end
Holdout(; fraction_train=0.7, shuffle=false) = Holdout(fraction_train, shuffle)

mutable struct CV <: ResamplingStrategy
    nfolds::Int
    parallel::Bool
    shuffle::Bool ## TODO: add seed/rng 
end
CV(; nfolds=6, parallel=true, shuffle=false) = CV(nfolds, parallel, shuffle)


## DIRECT EVALUATION METHODS 

# We first define an `evaluate!` to directly generate estimates of
# performance according to some strategy
# `s::S<:ResamplingStrategy`. (For tuning we later define a
# `Resampler{S}` object that wraps a model in a resampling strategy
# and a measure.)  We need an `evaluate!` for each strategy.

"""
    evaluate!(mach, resampling=CV(), measures=nothing, operation=predict, verbosity=1)

Estimate the performance of a machine `mach` using the specified
`resampling` (defaulting to 6-fold cross-validation) and
`measure`. In general this mutating operation preserves only
`mach.args` (the data stored in the machine).

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
                   measures=nothing, operation=predict, verbosity=1)

    if measures == nothing
        _measures = default_measure(mach.model)
        if _measures == nothing
            error("You need to specify measures=... ")
        end
    else
        _measures = measures
    end
    
    X = mach.args[1]
    y = mach.args[2]
    length(mach.args) == 2 || error("Multivariate targets not yet supported.")
    
    all =
        rows == nothing ? eachindex(y) : rows
    
    train, test = partition(all, resampling.fraction_train,
                            shuffle=resampling.shuffle)
    if verbosity > 0
            all == eachindex(y) ? "Resampling from all rows. " : "Resampling from a subset of all rows. "
        which_rows =
            all == eachindex(y) ? "Resampling from all rows. " : "Resampling from a subset of all rows. "
        @info "Evaluating using a holdout set. \n"*
        "fraction_train=$(resampling.fraction_train) \n"*
        "shuffle=$(resampling.shuffle) \n"*
        "measure=$_measure \n"*
        "operation=$operation \n"*
        "$which_rows"
    end

    fit!(mach, rows=train, verbosity=verbosity-1)
    yhat = operation(mach, selectrows(X, test))

    if !(_measures isa AbstractVector)
        return _measures(yhat, y[test])
    end

    res = Dict()
    for measure in _measures
        fitresult = measure(yhat, y[test])
        res[string(measure)] = fitresult
    end
    NamedTuple{Tuple(Symbol.(keys(res)))}(values(res))

end

# cv:
function evaluate!(mach::Machine, resampling::CV;
                   measures=nothing, operation=predict, rows=nothing, verbosity=1)

    if measures == nothing
        _measures = default_measure(mach.model)
        if _measures == nothing
            error("You need to specify measures=... ")
        end
    else
        _measures = measures
    end

    X = mach.args[1]
    y = mach.args[2]
    length(mach.args) == 2 || error("Multivariate targets not yet supported.")

    all =
        rows == nothing ? eachindex(y) : rows
    
    if verbosity > 0
        which_rows =
            all == eachindex(y) ? "Resampling from all rows. " : "Resampling from a subset of all rows. "
        @info "Evaluating using cross-validation. \n"*
        "nfolds=$(resampling.nfolds). \n"*
        "shuffle=$(resampling.shuffle) \n"*
        "measure=$_measure \n"*
        "operation=$operation \n"*
        "$which_rows"
    end

    n_samples = length(all)
    nfolds = resampling.nfolds
    
    if resampling.shuffle
        all = shuffle(all)
    end
    
    k = floor(Int,n_samples/nfolds)

    # function to return the measure for the fold `all[f:s]`:
    function get_measure(f, s)
        test = all[f:s] # TODO: replace with views?
        train = vcat(all[1:(f - 1)], all[(s + 1):end])
        fit!(mach; rows=train, verbosity=verbosity-1)
        yhat = operation(mach, selectrows(X, test))
        if !(_measures isa AbstractVector) 
            return _measures(yhat, y[test])
        else
            res = Dict()
            for measure in _measures
                fitresult = measure(yhat, y[test])
                res[string(measure)] = fitresult
            end
            return NamedTuple{Tuple(Symbol.(keys(res)))}(values(res))
        end
    end

    firsts = 1:k:((nfolds - 1)*k + 1) # itr of first `test` rows index
    seconds = k:k:(nfolds*k)          # itr of last `test` rows  index

    if resampling.parallel && nworkers() > 1
        ## TODO: progress meter for distributed case
        if verbosity > 0
            @info "Distributing cross-validation computation "*
            "among $(nworkers()) workers."
        end
        fitresult = @distributed vcat for n in 1:nfolds
            [get_measure(firsts[n], seconds[n])]
        end
    else
        if verbosity > 0
            p = Progress(nfolds + 1, dt=0, desc="Cross-validating: ",
                         barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
            next!(p)
            measures = [first((get_measure(firsts[n], seconds[n]), next!(p))) for n in 1:nfolds]
        else
            measures = [get_measure(firsts[n], seconds[n]) for n in 1:nfolds]
        end            
    end

    return measures

end


## RESAMPLER - A MODEL WRAPPER WITH `evaluate` OPERATION

# this is needed for the `TunedModel` `fit` defined in tuning.jl

mutable struct Resampler{S,M<:Supervised} <: Supervised{Any}
    model::M
    resampling::S # resampling strategy
    measure
    operation
end

MLJBase.package_name(::Type{<:Resampler}) = "MLJ"
    
Resampler(; model=ConstantRegressor(), resampling=Holdout(),
          measure=nothing, operation=predict) =
              Resampler(model, resampling, measure, operation) 

function MLJBase.fit(resampler::Resampler, verbosity::Int, X, y)

    if resampler.measure == nothing
        measure = default_measure(resampler.model)
        if measure == nothing
            error("You need to specify measure=... ")
        end
    else
        measure = resampler.measure
    end

    mach = machine(resampler.model, X, y)

    fitresult = evaluate!(mach, resampler.resampling;
                         measures=measure,
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

    if resampler.measure == nothing
        measure = default_measure(resampler.model)
        if measure == nothing
            error("You need to specify measures=... ")
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
                         measures=resampler.measure,
                         operation=resampler.operation,
                         verbosity=verbosity-1)
    
    report = NamedTuple

    return fitresult, cache, report
    
end

MLJBase.evaluate(model::Resampler, fitresult) = fitresult






    

    
