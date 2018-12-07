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
    metric
    function Holdout(fraction_train, metric)
        0 < fraction_train && fraction_train < 1 ||
            error("fraction_train must be between 0 and 1.")
        return new(fraction_train, metric)
    end
end
Holdout(; fraction_train=0.7, metric=rms) = Holdout(fraction_train, metric)

mutable struct CV <: ResamplingStrategy
    n_folds::Int
end
CV(; n_folds=6) = CV(n_folds)

mutable struct Resampler{S,M<:Supervised} <: Model
    model::M
    strategy::S
end
Resampler(;model=ConstantRegressor(), strategy=Holdout()) =
    Resampler(model, strategy) 

function fit(resampler::Resampler{Holdout}, verbosity, X, y)

    trainable_model = trainable(resampler.model, X, y)

    train, test = partition(eachindex(y), resampler.strategy.fraction_train)
    fit!(trainable_model, rows=train, verbosity=verbosity-1)
    yhat = predict(trainable_model, X[Rows, test])    
    fitresult = resampler.strategy.metric(y[test], yhat)

    # remember model and strategy for calls to update
    cache = (trainable_model,
             deepcopy(resampler.model),
             deepcopy(resampler.strategy),
             test)
    report = nothing

    return fitresult, cache, nothing
    
end

function update(resampler::Resampler{Holdout}, verbosity, fitresult, cache, X, y)

    trainable_model, oldmodel, oldstrategy, oldtest = cache

    if resampler.strategy == oldstrategy && resampler.model == oldmodel
        return fitresult, cache, nothing
    elseif resampler.strategy != oldstrategy
        train, test = partition(eachindex(y), resampler.strategy.fraction_train)
        fit!(trainable_model, rows=train, verbosity=verbosity-1)
    elseif resampler.model != oldmodel
        fit!(trainable_model, verbosity=verbosity-1)
        test = oldtest
    end
    yhat = predict(trainable_model, X[Rows, test])    
    fitresult = resampler.strategy.metric(y[test], yhat)

    cache = (trainable_model,
             deepcopy(resampler.model),
             deepcopy(resampler.strategy),
             test)
    
    report = nothing    
        
    return fitresult, cache, nothing

end

evaluate(model::Resampler{Holdout}, fitresult) = fitresult

             




## DIRECT EVALUTATING OF TRAINABLE MODELS

# # holdout evaluation:
# function evaluate(trainable_model, strategy::Holdout, metric, rows)
#     X, y = trainable_model.args
#     if rows == nothing
#         rows = eachindex(y)
#     end
#     train, test = partition(rows, strategy.fraction_train)
#     fit!(trainable_model, rows=train)
#     yhat = predict(trainable_model, X[Rows, test])
#     return metric(y, yhat)
# end

# # universal keyword version:
# evaluate(trainable_model::TrainableModel{<:Supervised};
#          strategy=Holdout,
#          metric=rms,
#          rows=nothing) =
#              evaluate(trainable_model, strategy, metric, rows)


    

    
