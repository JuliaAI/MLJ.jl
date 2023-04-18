let
    data = load_boston()
    schema(data)
    y, X = unpack(data, ==(:MedV))
    models(matching(X, y))
    doc("DecisionTreeClassifier", pkg="DecisionTree")
    mutable struct A <: Probabilistic
        k::Float64
    end
    A(; k=0.0) = A(k)

    MLJBase = MLJ.MLJBase
    I = MLJ.Distributions.I

    function MLJBase.fit(model::A, verbosity::Int, X, y)
        X = MLJBase.matrix(X)
        y = convert(Vector{Float64}, y)
        fitresult = (X'X + model.k * I) \ (X'y)
        cache = nothing
        report = nothing
        return fitresult, cache, report
    end

    function MLJBase.predict(model::A, fitresult, Xnew)
        MLJBase.matrix(Xnew) * fitresult
    end

    regressor = machine(A(), X, y)
    evaluate!(regressor; resampling=CV(; nfolds=2), measure=rms)
end

nothing
