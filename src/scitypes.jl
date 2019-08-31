function scitypes(X)
    s = schema(X)
    return (names=collect(s.names), scitypes=collect(s.scitypes))
end


## SUPERVISED

struct SupervisedModel{input_scitype, target_scitype, is_probabilistic} end

ScientificTypes.scitype(model::Deterministic, ::Val{:mlj}) =
    SupervisedModel{input_scitype(model),
                    target_scitype(model),
                    false}
                    
ScientificTypes.scitype(model::Probabilistic, ::Val{:mlj}) =
    SupervisedModel{input_scitype(model),
                    target_scitype(model),
                    true}
                    
function Base.getproperty(::SupervisedModel{input_scitype, target_scitype, is_probabilistic},
                          field::Symbol) where {input_scitype, target_scitype, is_probabilistic}
    if field === :input_scitype
        return input_scitype
    elseif field === :target_scitype
        return target_scitype
    elseif field === :is_probabilistic
        return is_probabilistic
    else
        throw(ArgumentError("Unsupported property. "))
    end
end

Base.propertynames(::SupervisedModel) = (:input_scitype, :target_scitype, :is_probabilistic)

_as_named_tuple(s::SupervisedModel) =
    NamedTuple{(:input_scitype, :target_scitype, :is_probabilistic)}((s.input_scitype, s.target_scitype, s.is_probabilistic))

function Base.show(io::IO, ::MIME"text/plain", S::Type{<:SupervisedModel})
    show(io, MIME("text/plain"), _as_named_tuple(S()))
end


## UNSUPERVISED

struct UnsupervisedModel{input_scitype, output_scitype} end

ScientificTypes.scitype(model::Unsupervised, ::Val{:mlj}) =
    UnsupervisedModel{input_scitype(model),
                      MLJBase.output_scitype(model)}

function Base.getproperty(::UnsupervisedModel{input_scitype, output_scitype},
                          field::Symbol) where {input_scitype, output_scitype}
    if field === :input_scitype
        return input_scitype
    elseif field === :output_scitype
        return output_scitype
    else
        throw(ArgumentError("Unsupported property. "))
    end
end

Base.propertynames(::UnsupervisedModel) = (:input_scitype, :output_scitype)

_as_named_tuple(s::UnsupervisedModel) =
    NamedTuple{(:input_scitype, :output_scitype)}(
        (s.input_scitype, s.output_scitype))

function Base.show(io::IO, ::MIME"text/plain", S::Type{<:UnsupervisedModel})
    show(io, MIME("text/plain"), _as_named_tuple(S()))
end

              
