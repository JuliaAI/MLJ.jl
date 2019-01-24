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

mutable struct Resampler{S,M<:Supervised} <: Model
    model::M
    resampling_strategy::S
    measure
    operation
end
Resampler(;model=RidgeRegressor(), resampling_strategy=Holdout(), measure=rms, operation=predict) =
    Resampler(model, resampling_strategy, measure, operation) 

function MLJBase.fit(resampler::Resampler{Holdout}, verbosity::Int, X, y)

    mach = machine(resampler.model, X, y)

    train, test = partition(eachindex(y), resampler.resampling_strategy.fraction_train)
    fit!(mach, rows=train, verbosity=verbosity-1)
    yhat = resampler.operation(mach, retrieve(X, Rows, test))    
    fitresult = resampler.measure(y[test], yhat)

    cache = mach
    report = nothing

    return fitresult, cache, report
    
end

function MLJBase.update(resampler::Resampler{Holdout}, verbosity::Int, fitresult, cache, X, y)

    mach = cache

    train, test = partition(eachindex(y), resampler.resampling_strategy.fraction_train)
    fit!(mach, rows=train, verbosity=verbosity-1)
    yhat = resampler.operation(mach, retrieve(X, Rows, test))    
    fitresult = resampler.measure(y[test], yhat)

    report = nothing

    return fitresult, cache, report
    
end

MLJBase.evaluate(model::Resampler{Holdout}, fitresult) = fitresult




    

    
