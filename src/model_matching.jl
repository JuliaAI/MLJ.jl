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

matching(X)       = Checker{false,false,scitype(X),missing}()
matching(X, y)    = Checker{true,false,scitype(X),scitype(y)}()
matching(X, y, w) = Checker{true,true,scitype(X),scitype(y)}()

(f::Checker{false,false,XS,missing})(model::MLJModels.ModelProxy) where XS =
    !(model.is_supervised) &&
    XS <: model.input_scitype

(f::Checker{true,false,XS,yS})(model::MLJModels.ModelProxy) where {XS,yS} =
    model.is_supervised &&
    XS <: model.input_scitype &&
    yS <: model.target_scitype

(f::Checker{true,true,XS,yS})(model::MLJModels.ModelProxy) where {XS,yS} =
    model.is_supervised &&
    model.supports_weights &&
    XS <: model.input_scitype &&
    yS <: model.target_scitype

(f::Checker)(name::String; pkg=nothing) = f(info(name, pkg=pkg))
(f::Checker)(realmodel::Model) = f(info(realmodel))

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



    

