# Note. `ModelProxy` is the type of a model's metadata entry (a named
# tuple). So, `info("PCA")` has this type, for example.


## BASIC IDEA

if false

    matching(model::MLJModels.ModelProxy, X) =
        !(model.is_supervised) && scitype(X) <: model.input_scitype

    matching(model::MLJModels.ModelProxy, X, y) =
        model.is_supervised &&
        scitype(X) <: model.input_scitype &&
        scitype(y) <: model.target_scitype

    matching(model::MLJModels.ModelProxy, X, y, w::AbstractVector{<:Real}) =
        model.is_supervised &&
        model.supports_weights &&
        scitype(X) <: model.input_scitype &&
        scitype(y) <: model.target_scitype

end


## IMPLEMENTATION


struct ModelChecker{is_supervised,
                  supports_weights,
                  input_scitype,
                    target_scitype} end

function Base.getproperty(::ModelChecker{is_supervised,
                                         supports_weights,
                                         input_scitype,
                                         target_scitype},
                          field::Symbol) where {is_supervised,
                                                supports_weights,
                                                input_scitype,
                                                target_scitype}
    if field === :is_supervised
        return is_supervised
    elseif field === :supports_weights
        return supports_weights
    elseif field === :input_scitype
        return input_scitype
    elseif field === :target_scitype
        return target_scitype
    else
        throw(ArgumentError("Unsupported property. "))
    end
end

Base.propertynames(::ModelChecker) =
    (:is_supervised, :supports_weights, :input_scitype, :target_scitype)

function _as_named_tuple(s::ModelChecker)
    names = propertynames(s)
    NamedTuple{names}(Tuple(getproperty(s, p) for p in names))
end

# function Base.show(io::IO, ::MIME"text/plain", S::ModelChecker)
#     show(io, MIME("text/plain"), _as_named_tuple(S))
# end

"""
   matching(model, X, y)

Returns `true` exactly when the registry metadata entry `model` is
supervised and admits inputs and targets with the scientific types of
`X` and `y`, respectively.

   matching(model, X)

Returns `true` exactly when `model` is unsupervised and admits inputs
with the scientific types of `X`.

    matching(model), matching(X, y), matching(X)

Curried versions of the preceding methods, i.e., `Bool`-valued
callable objects satisfying `matching(X, y)(model) = matching(model,
X, y)`, etc.

### Example

    models(matching(X))

Finds all unsupervised models compatible with input data `X`.

    models() do model
        matching(model, X, y) && model.prediction_type == :probabilistic
    end

Finds all supervised models compatible with input data `X` and target
data `y` and making probabilistic predictions.


See also [`models`](@ref)

"""
matching(X)       = ModelChecker{false,false,scitype(X),missing}()
matching(X, y)    = ModelChecker{true,false,scitype(X),scitype(y)}()
matching(X, y, w) = ModelChecker{true,true,scitype(X),scitype(y)}()

(f::ModelChecker{false,false,XS,missing})(model::MLJModels.ModelProxy) where XS =
    !(model.is_supervised) &&
    XS <: model.input_scitype

(f::ModelChecker{true,false,XS,yS})(model::MLJModels.ModelProxy) where {XS,yS} =
    model.is_supervised &&
    XS <: model.input_scitype &&
    yS <: model.target_scitype

(f::ModelChecker{true,true,XS,yS})(model::MLJModels.ModelProxy) where {XS,yS} =
    model.is_supervised &&
    model.supports_weights &&
    XS <: model.input_scitype &&
    yS <: model.target_scitype

(f::ModelChecker)(name::String; pkg=nothing) = f(info(name, pkg=pkg))
(f::ModelChecker)(realmodel::Model) = f(info(realmodel))

matching(model::MLJModels.ModelProxy, args...) = matching(args...)(model)
matching(name::String, args...; pkg=nothing) =
    matching(info(name, pkg=pkg), args...)
matching(realmodel::Model, args...) = matching(info(realmodel), args...)


## DUAL NOTION

struct DataChecker
    model::MLJModels.ModelProxy
end

matching(model::MLJModels.ModelProxy) = DataChecker(model)
matching(name::String; pkg=nothing) = DataChecker(info(name, pkg=pkg))
matching(realmodel::Model) = matching(info(realmodel))

(f::DataChecker)(args...) = matching(f.model, args...)



    

