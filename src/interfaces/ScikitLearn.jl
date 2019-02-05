module ScikitLearn_
#> export the new models you're going to define (and nothing else):
export SVMClassifier, SVMRegressor
#> for all Supervised models:
import MLJBase

#> for all classifiers:
using CategoricalArrays

#> import package:
import ScikitLearn: @sk_import
@sk_import svm: SVC
@sk_import svm: SVR


mutable struct SVMClassifier{Any} <: MLJBase.Deterministic{Any}
    C::Float64 
    kernel::Union{String,Function}
    degree::Int
    gamma::Union{Float64,String}
    coef0::Float64
    shrinking::Bool
    probability::Bool
    tol::Float64
    cache_size::Float64
    max_iter::Int
    decision_function_shape::String
    random_state
end

# constructor:
#> all arguments are kwargs with a default value
function SVMClassifier(
    ;C=1.0
    ,kernel="rbf"
    ,degree=3
    ,gamma="auto"
    ,coef0=0.0
    ,shrinking=true
    ,probability=false
    ,tol=1e-3
    ,cache_size=200
    ,max_iter=-1
    ,decision_function_shape="ovr"
    ,random_state=nothing)

    model = SVMClassifier{Any}(
        C
        , kernel
        , degree
        , gamma
        , coef0
        , shrinking
        , probability
        , tol
        , cache_size
        , max_iter
        , decision_function_shape
        , random_state
        )

    message = MLJBase.clean!(model)       #> future proof by including these 
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end


function MLJBase.fit(model::SVMClassifier{Any}
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X::Array{<:Real,2}
             , y)

    #decoder = MLJBase.CategoricalDecoder(y)
    #y_plain = MLJBase.transform(decoder, y)
    
    cache = SVC(C=model.c,
            kernel=model.kernel,
            degree=model.degree,
            coef0=model.coef0,
            shrinking=model.shrinking,
            probability=model.probability,
            tol=model.tol,
            cache_size=model.cache_size,
            max_iter=model.max_iter,
            decision_function_shape=model.decision_function_shape,
            random_state=model.random_state
    )
    
    fitresult = ScikitLearn.fit!(cache,X,y)

    report = nothing
    
    return fitresult, cache, report 

end


mutable struct SVMRegressor{Any} <: MLJBase.Deterministic{Any}
    C::Float64 
    kernel::Union{String,Function}
    degree::Int
    gamma::Union{Float64,String}
    coef0::Float64
    shrinking::Bool
    probability::Bool
    tol::Float64
    cache_size::Float64
    max_iter::Int
    epsilon::Float64
end

# constructor:
#> all arguments are kwargs with a default value
function SVMRegressor(
    ;C=1.0
    ,kernel="rbf"
    ,degree=3
    ,gamma="auto"
    ,coef0=0.0
    ,shrinking=true
    ,probability=false
    ,tol=1e-3
    ,cache_size=200
    ,max_iter=-1
    ,epsilon=0.1)

    model = SVMRegressor{Any}(
        C
        , kernel
        , degree
        , gamma
        , coef0
        , shrinking
        , probability
        , tol
        , cache_size
        , max_iter
        , epsilon)

    message = MLJBase.clean!(model)       #> future proof by including these 
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end


function MLJBase.fit(model::SVMRegressor{Any}
             , verbosity::Int   #> must be here (and typed) even if not used (as here)
             , X::Array{<:Real,2}
             , y)

    #decoder = MLJBase.CategoricalDecoder(y)
    #y_plain = MLJBase.transform(decoder, y)
    
    cache = SVR(C=model.c,
            kernel=model.kernel,
            degree=model.degree,
            coef0=model.coef0,
            shrinking=model.shrinking,
            probability=model.probability,
            tol=model.tol,
            cache_size=model.cache_size,
            max_iter=model.max_iter,
            epsilon=model.epsilon)
    
    fitresult = ScikitLearn.fit!(cache,X,y)

    report = nothing
    
    return fitresult, cache, report 

end





#> method to coerce generic data into form required by fit:

function MLJBase.predict(model::Union{SVMClassifier{Any},SVMRegressor{Any}}
                     , fitresult
                     , Xnew) 
    prediction = ScitkitLearn.predict(fitresult,Xnew)
    return prediction
end

# metadata:
#MLJBase.package_name(::Type{<:SVMClassifier}) = "ScitkitLearn"
#MLJBase.package_uuid(::Type{<:SVMClassifier}) = "7"
#MLJBase.is_pure_julia(::Type{<:SVMClassifier}) = :no
#MLJBase.inputs_can_be(::Type{<:SVMClassifier}) = [:numeric, ]
#MLJBase.target_kind(::Type{<:SVMClassifier}) = :
#MLJBase.target_quantity(::Type{<:SVMClassifier}) = :

end # module


## EXPOSE THE INTERFACE

using .ScikitLearn_
export SVMClassifier,SVMRegressor    
