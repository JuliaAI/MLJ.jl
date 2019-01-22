module GaussianProcesses_

export GPClassifier

import MLJBase

using CategoricalArrays

import GaussianProcesses

const GP = GaussianProcesses

GPClassifierFitResultType{T} =
    Tuple{GP.GPE,
    MLJBase.CategoricalDecoder{UInt32,T,1,UInt32}}

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
            , X::Matrix{Float64}
            , y::CategoricalVector{T}) where {T,T2,M,K}

    T == T2 || throw(ErrorException("Type, $T, of target incompatible "*
                                    "with type, $T2, of $model."))

    decoder = MLJBase.CategoricalDecoder(y, eltype=Int)
    y_plain = MLJBase.transform(decoder, y)


    if VERSION < v"1.0"
        XT = collect(X')
        yP = convert(Vector{Float64}, y_plain)
        gp = GP.GPE(XT
                  , yP
                  , model.mean
                  , model.kernel)

        GP.fit!(gp, XT, yP)
    else
        gp = GP.GPE(X'
                  , y_plain
                  , model.mean
                  , model.kernel)
        GP.fit!(gp, X', y_plain)
    end

    fitresult = (gp, decoder)

    cache = nothing
    report = nothing

    return fitresult, cache, report
end

MLJBase.coerce(model::GPClassifier, Xtable) = MLJBase.matrix(Xtable)

function MLJBase.predict(model::GPClassifier{T}
                       , fitresult
                       , Xnew) where T

    gp, decoder = fitresult

    nlevels = length(decoder.pool.levels)
    pred = GP.predict_y(gp, Xnew')[1] # Float
    # rounding with clamping between 1 and nlevels
    pred_rc = clamp.(round.(Int, pred), 1, nlevels)

    return MLJBase.inverse_transform(decoder, pred_rc)
end

# metadata:
MLJBase.package_name(::Type{<:GPClassifier}) = "GaussianProcesses"
MLJBase.package_uuid(::Type{<:GPClassifier}) = "891a1506-143c-57d2-908e-e1f8e92e6de9"
MLJBase.is_pure_julia(::Type{<:GPClassifier}) = :yes
MLJBase.inputs_can_be(::Type{<:GPClassifier}) = [:numeric, ]
MLJBase.target_kind(::Type{<:GPClassifier}) = :multiclass
MLJBase.target_quantity(::Type{<:GPClassifier}) = :univariate

end # module

using .GaussianProcesses_
export GPClassifier
