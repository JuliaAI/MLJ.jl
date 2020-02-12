## SUPERVISED

const MST = MLJScientificTypes # only used in this file

struct SupervisedScitype{input_scitype, target_scitype, prediction_type} end

MST.scitype(model::Deterministic, ::MST.MLJ) =
    SupervisedScitype{input_scitype(model),
                    target_scitype(model),
                    :deterministic}

MST.scitype(model::Probabilistic, ::MST.MLJ) =
    SupervisedScitype{input_scitype(model),
                    target_scitype(model),
                    :probabilistic}

MST.scitype(model::Interval, ::MST.MLJ) =
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

MST.scitype(model::Unsupervised, ::MST.MLJ) =
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


## MEASURES

struct MeasureScitype{target_scitype,
               prediction_type,
               orientation,
               reports_each_observation,
               is_feature_dependent,
               supports_weights} end

MST.scitype(measure, ::MST.MLJ, ::Val{:measure}) =
    MeasureScitype{target_scitype(measure),
               prediction_type(measure),
               orientation(measure),
               reports_each_observation(measure),
               is_feature_dependent(measure),
               supports_weights(measure)}

function Base.getproperty(::MeasureScitype{target_scitype,
               prediction_type,
               orientation,
               reports_each_observation,
               is_feature_dependent,
               supports_weights},
                          field::Symbol) where {target_scitype,
                                                prediction_type,
                                                orientation,
                                                reports_each_observation,
                                                is_feature_dependent,
                                                supports_weights}
    if field === :target_scitype
        return target_scitype
    elseif field === :prediction_type
        return prediction_type
    elseif field === :orientation
        return orientation
    elseif field === :reports_each_observation
        return reports_each_observation
    elseif field === :is_feature_dependent
        return is_feature_dependent
    elseif field === :supports_weights
        return supports_weights
    else
        throw(ArgumentError("Unsupported property. "))
    end
end

# crashes julia:
# MLJBase.getproperty(M::Type{<:MeasureScitype}, p::Symbol) =
#     getproperty(M(), p)

Base.propertynames(::MeasureScitype) =
    (:target_scitype, :prediction_type, :orientation,
     :reports_each_observation, :is_feature_dependent, :supports_weights)

function _as_named_tuple(m::MeasureScitype)
    names = propertynames(m)
    return NamedTuple{names}(Tuple([getproperty(m, p) for p in names]))
end

function Base.show(io::IO, ::MIME"text/plain", M::MeasureScitype)
      show(io, MIME("text/plain"), _as_named_tuple(M))
end
