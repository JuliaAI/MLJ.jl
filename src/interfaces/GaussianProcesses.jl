module GaussianProcesses_

export GPClassifier

import MLJBase

using CategoricalArrays

import GaussianProcesses

const GP = GaussianProcesses

# here T is target type:
const CD{T,C} = MLJBase.CategoricalDecoder{Int,false,T,1,UInt32,C}
const GPClassifierFitResultType{T} =
    Tuple{GP.GPE,     # TODO: make this a concrete type for ensembling efficiency
          Union{CD{T,CategoricalValue{T,UInt32}},
                CD{T,CategoricalString{UInt32}}}}

mutable struct GPClassifier{T, M<:GP.Mean, K<:GP.Kernel} <: MLJBase.Deterministic{GPClassifierFitResultType{T}}
    target_type::Type{T} # target is CategoricalArray{target_type}
    mean::M
    kernel::K
end

function GPClassifier(
    ; target_type=Int
    , mean=GP.MeanZero()
    , kernel=GP.SE(0.0,1.0)) # binary

    model = GPClassifier(
        target_type
        , mean
        , kernel)

    message = MLJBase.clean!(model)
    isempty(message) || @warn message

    return model
end

# function MLJBase.clean!

function MLJBase.fit(model::GPClassifier{T2,M,K}
            , verbosity::Int
            , X
            , y::CategoricalVector{T}) where {T,T2,M,K}

    Xmatrix = MLJBase.matrix(X)
    
    T == T2 || throw(ErrorException("Type, $T, of target incompatible "*
                                    "with type, $T2, of $model."))

    decoder = MLJBase.CategoricalDecoder(y, Int)
    y_plain = MLJBase.transform(decoder, y)


    if VERSION < v"1.0"
        XT = collect(transpose(Xmatrix))
        yP = convert(Vector{Float64}, y_plain)
        gp = GP.GPE(XT
                  , yP
                  , model.mean
                  , model.kernel)

        GP.fit!(gp, XT, yP)
    else
        gp = GP.GPE(transpose(Xmatrix)
                  , y_plain
                  , model.mean
                  , model.kernel)
        GP.fit!(gp, transpose(Xmatrix), y_plain)
    end

    fitresult = (gp, decoder)

    cache = nothing
    report = nothing

    return fitresult, cache, report
end

function MLJBase.predict(model::GPClassifier{T}
                       , fitresult
                       , Xnew) where T

    Xmatrix = MLJBase.matrix(Xnew)
    
    gp, decoder = fitresult

    nlevels = length(decoder.pool.levels)
    pred = GP.predict_y(gp, transpose(Xmatrix))[1] # Float
    # rounding with clamping between 1 and nlevels
    pred_rc = clamp.(round.(Int, pred), 1, nlevels)

    return MLJBase.inverse_transform(decoder, pred_rc)
end

# metadata:
MLJBase.load_path(::Type{<:GPClassifier}) = "MLJ.GPClassifier" # lazy-loaded from MLJ
MLJBase.package_name(::Type{<:GPClassifier}) = "GaussianProcesses"
MLJBase.package_uuid(::Type{<:GPClassifier}) = "891a1506-143c-57d2-908e-e1f8e92e6de9"
MLJBase.package_url(::Type{<:GPClassifier}) = "https://github.com/STOR-i/GaussianProcesses.jl"
MLJBase.is_pure_julia(::Type{<:GPClassifier}) = :yes
MLJBase.input_kinds(::Type{<:GPClassifier}) = [:continuous, ]
MLJBase.output_kind(::Type{<:GPClassifier}) = :ordered_factor_finite
MLJBase.output_quantity(::Type{<:GPClassifier}) = :univariate

end # module

using .GaussianProcesses_
export GPClassifier
