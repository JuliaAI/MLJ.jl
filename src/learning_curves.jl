## LEARNING CURVES

"""
    curve = learning_curve!(mach; resolution=30,
                                  resampling=Holdout(),
                                  measure=rms,
                                  operation=predict,
                                  range=nothing,
                                  n=1)

Given a supervised machine `mach`, returns a named tuple of objects
suitable for generating a plot of  performance measurements, as a function
of the single hyperparameter specified in `range`. The tuple `curve`
has the following keys: `:parameter_name`, `:parameter_scale`,
`:parameter_values`, `:measurements`.

For `n > 1`, multiple curves are computed, and the value of
`curve.measurements` is an array, one column for each run. This is
useful in the case of models with indeterminate fit-results, such as a
random forest.

````julia
X, y = @load_boston;
atom = @load RidgeRegressor pkg=MultivariateStats
ensemble = EnsembleModel(atom=atom, n=1000)
mach = machine(ensemble, X, y)
r_lambda = range(ensemble, :(atom.lambda), lower=10, upper=500, scale=:log10)
curve = MLJ.learning_curve!(mach; range=r_lambda, resampling=CV(), measure=mav)
using Plots
plot(curve.parameter_values,
     curve.measurements,
     xlab=curve.parameter_name,
     xscale=curve.parameter_scale,
     ylab = "CV estimate of RMS error")
````

If using a `Holdout` `resampling` strategy, and the specified
hyperparameter is the number of iterations in some iterative model
(and that model has an appropriately overloaded `MLJBase.update`
method) then training is not restarted from scratch for each increment
of the parameter, ie the model is trained progressively.

````julia
atom.lambda=200
r_n = range(ensemble, :n, lower=1, upper=250)
curves = MLJ.learning_curve!(mach; range=r_n, verbosity=0, n=5)
plot(curves.parameter_values, 
     curves.measurements, 
     xlab=curves.parameter_name,
     ylab="Holdout estimate of RMS error")
````

"""
function learning_curve!(mach::Machine{<:Supervised};
                         resolution=30, resampling=Holdout(),
                         measure=nothing,
                         operation=predict,
                         range=nothing, verbosity=1, n=1)

    if measure == nothing
        measure = default_measure(mach.model)
        verbosity < 1 ||
            @info "No measure specified. Using measure=$measure. "
    end

    range !== nothing || error("No param range specified. Use range=... ")

    tuned_model = TunedModel(model=mach.model, ranges=range,
                             tuning=Grid(resolution=resolution),
                             resampling=resampling,
                             operation=operation,
                             measure=measure,
                             full_report=true, train_best=false)

    tuned = machine(tuned_model, mach.args...)

    measurements = reduce(hcat, [(fit!(tuned, verbosity=verbosity, force=true);
                                  tuned.report.measurements) for c in 1:n])
    report = tuned.report
    parameter_name=report.parameter_names[1]
    parameter_scale=report.parameter_scales[1]
    parameter_values=[report.parameter_values[:, 1]...]
    measurements_ = (n == 1) ? [measurements...] : measurements

    return (parameter_name=parameter_name,
            parameter_scale=parameter_scale,
            parameter_values=parameter_values,
            measurements = measurements_)
end

"""
    learning_curve(model::Supervised, args...; kwargs...)

Plot a learning curve (or curves) without first constructing a
machine. Equivalent to `learing_curve!(machine(model, args...);
kwargs...)

See [learning_curve!](@ref)

"""
learning_curve(model::Supervised, args...; kwargs...) =
    learning_curve!(machine(model, args...); kwargs...)
