# fall-back for updating learning networks exported as models:
function update(model::Supervised{Node}, verbosity, fitresult, cache, args...)
    fit!(fitresult; verbosity=verbosity)
    return fitresult, cache, nothing
end

# fall-back for predicting on learning networkds exported as models
MLJ.predict(composite::Supervised{Node}, verbosity, fitresult, Xnew) =
    fitresult(coerce(composite, Xnew))


"""
    SimpleCompositeRegressor(;regressor=ConstantRegressor(), 
                              transformer_X=FeatureSelector(),
                              transformer_y=UnivariateStandardizer())

Construct a composite model which wraps `regressor` in a
pre-transformation of the input features, defined by the unsuperived
learner `transformer_X`, and a pre-transformation of the target,
defined by the unsupervised learner `transformer_y`, which is also
used to inverse-transform the regressor predictions.

"""
mutable struct SimpleCompositeRegressor{L<:Supervised,
                             TX<:Unsupervised,
                             Ty<:Unsupervised} <: Supervised{Node}
    regressor::L
    transformer_X::TX
    transformer_y::Ty
end

SimpleCompositeRegressor(; regressor=ConstantRegressor(), 
                          transformer_X=FeatureSelector(),
                          transformer_y=UnivariateStandardizer()) =
                              SimpleCompositeRegressor(regressor,
                                                       transformer_X,
                                                       transformer_y)

function MLJ.fit(composite::SimpleCompositeRegressor, verbosity, Xtrain, ytrain)
    X = source(Xtrain) # instantiates a source node
    y = source(ytrain)
    t_X = machine(composite.transformer_X, X)
    t_y = machine(composite.transformer_y, y)
    Xt = transform(t_X, X)
    yt = transform(t_y, y)
    l = machine(composite.regressor, Xt, yt)
    zhat = predict(l, Xt)
    yhat = inverse_transform(t_y, zhat)
    fit!(yhat, verbosity=verbosity)
    fitresult = yhat
    report = l.report
    cache = l
    return fitresult, cache, report
end

MLJ.predict(composite::SimpleCompositeRegressor, fitresult, Xnew) = fitresult(Xnew)

