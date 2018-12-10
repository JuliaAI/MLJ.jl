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
end
CV(; n_folds=6) = CV(n_folds)

mutable struct Resampler{S,M<:Supervised} <: Model
    model::M
    tuning::S
    measure
end
Resampler(;model=ConstantRegressor(), tuning=Holdout(), measure=rms) =
    Resampler(model, tuning, measure) 

function fit(resampler::Resampler{Holdout}, verbosity, X, y)

    trainable_model = trainable(resampler.model, X, y)

    train, test = partition(eachindex(y), resampler.tuning.fraction_train)
    fit!(trainable_model, rows=train, verbosity=verbosity-1)
    yhat = predict(trainable_model, X[Rows, test])    
    fitresult = resampler.measure(y[test], yhat)

    # remember model and tuning stragegy for calls to update
    cache = nothing
    report = nothing

    return fitresult, cache, report
    
end

evaluate(model::Resampler{Holdout}, fitresult) = fitresult


## DIRECT EVALUTATING OF TRAINABLE MODELS

# # holdout evaluation:
# function evaluate(trainable_model, tuning::Holdout, measure, rows)
#     X, y = trainable_model.args
#     if rows == nothing
#         rows = eachindex(y)
#     end
#     train, test = partition(rows, tuning.fraction_train)
#     fit!(trainable_model, rows=train)
#     yhat = predict(trainable_model, X[Rows, test])
#     return measure(y[test], yhat)
# end

# # universal keyword version:
# evaluate(trainable_model::TrainableModel{<:Supervised};
#          tuning=Holdout,
#          measure=rms,
#          rows=nothing) =
#              evaluate(trainable_model, tuning, measure, rows)


    

    
