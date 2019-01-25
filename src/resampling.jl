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
    n_folds::Int
    is_parallel::Bool
end
CV(; n_folds=6) = CV(n_folds)


## RESAMPLER - MODEL WITH `evaluate` OPERATION

mutable struct Resampler{S,M<:Supervised} <: Model
    model::M
    resampling_strategy::S
    measure
    operation
end
Resampler(;model=RidgeRegressor(), resampling_strategy=Holdout(), measure=rms, operation=predict) =
    Resampler(model, resampling_strategy, measure, operation) 


## DIRECT EVALUATION METHODS 

# We first define an `evaluate` to directly generate estimates of
# performance according to some strategy `s::S<:ResamplingStrategy`,
# *without* first building and fitting a `Resampler{S}` object. We
# need one for each strategy.

function MLJBase.evaluate(mach::Machine, strategy::Holdout;
                          measure=rms, operation=predict, verbosity=1)

    X = mach.args[1]
    y = mach.args[2]
    length(mach.args) == 2 || error("Multivariate targets not yet supported.")
    
    train, test = partition(eachindex(y), strategy.fraction_train)
    fit!(mach, rows=train, verbosity=verbosity-1)
    yhat = operation(mach, retrieve(X, Rows, test))    
    fitresult = measure(y[test], yhat)

end

function MLJBase.fit(resampler::Resampler{Holdout}, verbosity::Int, X, y)

    mach = machine(resampler.model, X, y)

    fitresult = evaluate(mach, resampler.resampling_strategy;
                         measure=resampler.measure,
                         operation=resampler.operation,
                         verbosity=verbosity-1)
    
    cache = mach
    report = nothing

    return fitresult, cache, report
    
end

function MLJBase.update(resampler::Resampler{Holdout}, verbosity::Int, fitresult, cache, X, y)

    mach = cache

    fitresult = evaluate(mach, resampler.resampling_strategy;
                         measure=resampler.measure,
                         operation=resampler.operation,
                         verbosity=verbosity-1)
    
    report = nothing

    return fitresult, cache, report
    
end

MLJBase.evaluate(model::Resampler{Holdout}, fitresult) = fitresult

#####





    

    
