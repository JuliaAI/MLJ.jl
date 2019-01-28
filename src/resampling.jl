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
    function Holdout(fraction_train)
        0 < fraction_train && fraction_train < 1 ||
            error("fraction_train must be between 0 and 1.")
        return new(fraction_train)
    end
end
Holdout(; fraction_train=0.7) = Holdout(fraction_train)

mutable struct CV <: ResamplingStrategy
    nfolds::Int
    parallel::Bool
    shuffle::Bool ## TODO: add seed/rng 
end
CV(; nfolds=6, parallel=true, shuffle=false) = CV(nfolds, parallel, shuffle)


## DIRECT EVALUATION METHODS 

# We first define an `evaluate` to directly generate estimates of
# performance according to some strategy `s::S<:ResamplingStrategy`,
# without first building and fitting a `Resampler{S}` object defined
# later. We need an `evaluate` for each strategy.

# MLJBase.evaluate(mach::Machine; resampling_strategy=Holdout()) =
#     evaluate(mach, resampling_strategy)

# holdout:
function MLJBase.evaluate(mach::Machine, strategy::Holdout;
                          measure=rms, operation=predict, verbosity=1)

    X = mach.args[1]
    y = mach.args[2]
    length(mach.args) == 2 || error("Multivariate targets not yet supported.")
    
    train, test = partition(eachindex(y), strategy.fraction_train)
    fit!(mach, rows=train, verbosity=verbosity-1)
    yhat = operation(mach, selectrows(X, test))    
    fitresult = measure(y[test], yhat)

end

# cv:
function MLJBase.evaluate(mach::Machine, strategy::CV;
                          measure=rms, operation=predict, verbosity=1)

    X = mach.args[1]
    y = mach.args[2]
    length(mach.args) == 2 || error("Multivariate targets not yet supported.")

    n_samples = length(y)
    nfolds = strategy.nfolds
    
    if strategy.shuffle
        rows = shuffle(eachindex(y))
    else
        rows = eachindex(y)
    end
    
    k = floor(Int,n_samples/nfolds)

    # function to return the measure for the fold `rows[f:s]`:
    function get_measure(f, s)
        test = rows[f:s] # TODO: replace with views?
        train = vcat(rows[1:(f - 1)], rows[(s + 1):end])
        fit!(mach, rows=train; verbosity=verbosity-1)
        yhat = operation(mach, selectrows(X, test))    
        return measure(y[test], yhat)
    end

    firsts = 1:k:((nfolds - 1)*k + 1) # itr of first `test` rows index
    seconds = k:k:(nfolds*k)          # itr of last `test` rows  index

    if strategy.parallel && nworkers() > 1
        ## TODO: progress meter for distributed case
        if verbosity > 0
            @info "Distributing cross-validation computation "*
            "among $(nworkers()) workers."
        end
        fitresult = @distributed vcat for n in 1:nfolds
            [get_measure(firsts[n], seconds[n])]
        end
    else
        p = Progress(nfolds + 1, dt=0, desc="Cross-validating: ",
                     barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
        next!(p)
        measures = [first((get_measure(firsts[n], seconds[n]), next!(p))) for n in 1:nfolds]
    end

    return measures

end


## RESAMPLER - A MODEL WRAPPER WITH `evaluate` OPERATION

mutable struct Resampler{S,M<:Supervised} <: Model
    model::M
    resampling_strategy::S
    measure
    operation
end
Resampler(;model=RidgeRegressor(), resampling_strategy=Holdout(),
          measure=rms, operation=predict) =
              Resampler(model, resampling_strategy, measure, operation) 

function MLJBase.fit(resampler::Resampler, verbosity::Int, X, y)

    mach = machine(resampler.model, X, y)

    fitresult = evaluate(mach, resampler.resampling_strategy;
                         measure=resampler.measure,
                         operation=resampler.operation,
                         verbosity=verbosity-1)
    
    cache = (mach, deepcopy(resampler.resampling_strategy))
    report = nothing

    return fitresult, cache, report
    
end

# in special case of holdout, we can reuse the underlying model's
# machine, provided the training_fraction has not changed:
function MLJBase.update(resampler::Resampler{Holdout}, verbosity::Int, fitresult, cache, X, y)

    old_mach, old_resampling_strategy = cache

    if old_resampling_strategy.fraction_train == resampler.resampling_strategy.fraction_train
        mach = old_mach
    else
        mach = machine(resampler.model, X, y)
        cache = (mach, deepcopy(resampler.resampling_strategy))
    end

    fitresult = evaluate(mach, resampler.resampling_strategy;
                         measure=resampler.measure,
                         operation=resampler.operation,
                         verbosity=verbosity-1)
    
    report = nothing

    return fitresult, cache, report
    
end

MLJBase.evaluate(model::Resampler, fitresult) = fitresult






    

    
