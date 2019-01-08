# fall-back for updating learning networks exported as models:
function update(model::Supervised{Node}, verbosity, fitresult, cache, args...)
    fit!(fitresult; verbosity=verbosity)
    return fitresult, cache, nothing
end

# fall-back for predicting on learning networks exported as models
MLJ.predict(composite::Supervised{Node}, verbosity, fitresult, Xnew) =
    fitresult(Xnew)


"""
    SimpleComposite(;regressor=ConstantRegressor(), 
                              transformer=FeatureSelector())

Construct a composite model consisting of a transformer
(`Unsupervised` model) followed by a `Supervised` model.

"""
mutable struct SimpleComposite{L<:Supervised,
                             T<:Unsupervised} <: Supervised{Node}
    model::L
    transformer::T
    
end

function SimpleComposite(; model=ConstantRegressor(), 
                          transformer=FeatureSelector())

    composite =  SimpleComposite(model, transformer)

    message = MLJ.clean!(composite)
    isempty(message) || @warn message

    return composite

end

function MLJ.fit(composite::SimpleComposite, verbosity, Xtrain, ytrain)
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

MLJ.predict(composite::SimpleComposite, fitresult, Xnew) = fitresult(Xnew)

