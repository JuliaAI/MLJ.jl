const SupervisedNetwork = Union{DeterministicNetwork,ProbabilisticNetwork}

# to suppress inclusion in models():
MLJBase.is_wrapper(::Type{DeterministicNetwork}) = true
MLJBase.is_wrapper(::Type{ProbabilisticNetwork}) = true

# fall-back for updating learning networks exported as models:
function MLJBase.update(model::SupervisedNetwork, verbosity, fitresult, cache, args...)
    fit!(fitresult; verbosity=verbosity)
    return fitresult, cache, nothing
end

# fall-back for predicting on learning networks exported as models
MLJBase.predict(composite::SupervisedNetwork, fitresult, Xnew) =
    fitresult(Xnew)


"""
    SimpleDeterministicCompositeModel(;regressor=ConstantRegressor(), 
                              transformer=FeatureSelector())

Construct a composite model consisting of a transformer
(`Unsupervised` model) followed by a `Deterministic` model. Mainly
intended for internal testing .

"""
mutable struct SimpleDeterministicCompositeModel{L<:Deterministic,
                             T<:Unsupervised} <: DeterministicNetwork
    model::L
    transformer::T
    
end

function SimpleDeterministicCompositeModel(; model=DeterministicConstantRegressor(), 
                          transformer=FeatureSelector())

    composite =  SimpleDeterministicCompositeModel(model, transformer)

    message = MLJ.clean!(composite)
    isempty(message) || @warn message

    return composite

end

MLJBase.is_wrapper(::Type{<:SimpleDeterministicCompositeModel}) = true

function MLJBase.fit(composite::SimpleDeterministicCompositeModel, verbosity::Int, Xtrain, ytrain)
    X = source(Xtrain) # instantiates a source node
    y = source(ytrain)

    t = machine(composite.transformer, X)
    Xt = transform(t, X)

    l = machine(composite.model, Xt, y)
    yhat = predict(l, Xt)

    fit!(yhat, verbosity=verbosity)
    fitresult = yhat
    report = l.report
    cache = l
    return fitresult, cache, report
end

# MLJBase.predict(composite::SimpleDeterministicCompositeModel, fitresult, Xnew) = fitresult(Xnew)

MLJBase.load_path(::Type{<:SimpleDeterministicCompositeModel}) = "MLJ.SimpleDeterministicCompositeModel"
MLJBase.package_name(::Type{<:SimpleDeterministicCompositeModel}) = "MLJ"
MLJBase.package_uuid(::Type{<:SimpleDeterministicCompositeModel}) = ""
MLJBase.package_url(::Type{<:SimpleDeterministicCompositeModel}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:SimpleDeterministicCompositeModel}) = true
# MLJBase.input_scitype_union(::Type{<:SimpleDeterministicCompositeModel}) = 
# MLJBase.target_scitype_union(::Type{<:SimpleDeterministicCompositeModel}) = 

