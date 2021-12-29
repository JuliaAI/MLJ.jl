## SUPERVISED

# This implementation of scitype for models and measures is highly
# experimental and not part of the public API

# only used in this file
const DefaultConvention = ScientificTypes.DefaultConvention
const STB = ScientificTypes.ScientificTypesBase

struct SupervisedScitype{input_scitype, target_scitype, prediction_type} end

STB.scitype(model::Deterministic, ::DefaultConvention) =
    SupervisedScitype{input_scitype(model),
                    target_scitype(model),
                    :deterministic}

STB.scitype(model::Probabilistic, ::DefaultConvention) =
    SupervisedScitype{input_scitype(model),
                    target_scitype(model),
                    :probabilistic}

STB.scitype(model::Interval, ::DefaultConvention) =
    SupervisedScitype{input_scitype(model),
                    target_scitype(model),
                    :interval}

function Base.getproperty(::SupervisedScitype{input_scitype, target_scitype, prediction_type},
                          field::Symbol) where {input_scitype, target_scitype, prediction_type}
    if field === :input_scitype
        return input_scitype
    elseif field === :target_scitype
        return target_scitype
    elseif field === :prediction_type
        return prediction_type
    else
        throw(ArgumentError("Unsupported property. "))
    end
end

# crashes julia:
# MLJBase.getproperty(S::Type{<:SupervisedScitype}, p::Symbol) =
#     getproperty(S(), p)

Base.propertynames(::SupervisedScitype) = (:input_scitype, :target_scitype, :prediction_type)

_as_named_tuple(s::SupervisedScitype) =
    NamedTuple{(:input_scitype, :target_scitype, :prediction_type)}((s.input_scitype, s.target_scitype, s.prediction_type))

function Base.show(io::IO, ::MIME"text/plain", S::SupervisedScitype)
    show(io, MIME("text/plain"), _as_named_tuple(S))
end


## UNSUPERVISED

struct UnsupervisedScitype{input_scitype, output_scitype} end

STB.scitype(model::Unsupervised, ::DefaultConvention) =
    UnsupervisedScitype{input_scitype(model),
                      MLJBase.output_scitype(model)}

function Base.getproperty(::UnsupervisedScitype{input_scitype, output_scitype},
                          field::Symbol) where {input_scitype, output_scitype}
    if field === :input_scitype
        return input_scitype
    elseif field === :output_scitype
        return output_scitype
    else
        throw(ArgumentError("Unsupported property. "))
    end
end

# crashes julia:
# MLJBase.getproperty(U::Type{<:UnsupervisedScitype}, p::Symbol) =
#     getproperty(U(), p)

Base.propertynames(::UnsupervisedScitype) = (:input_scitype, :output_scitype)

_as_named_tuple(s::UnsupervisedScitype) =
    NamedTuple{(:input_scitype, :output_scitype)}(
        (s.input_scitype, s.output_scitype))

function Base.show(io::IO, ::MIME"text/plain", S::UnsupervisedScitype)
    show(io, MIME("text/plain"), _as_named_tuple(S))
end


