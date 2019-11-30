abstract type TuningStrategy <: MLJ.MLJType end
const ParameterName=Union{Symbol,Expr}

"""
    Grid(resolution=10, parallel=true)

Define a grid-based hyperparameter tuning strategy, using the
specified `resolution` for numeric hyperparameters. For use with a
`TunedModel` object.

Individual hyperparameter resolutions can also be specified, as in

    Grid(resolution=[:n => r1, :(atom.max_depth) => r2])

where `r1` and `r2` are `NumericRange` objects.

See also [TunedModel](@ref), [range](@ref).

"""
mutable struct Grid <: TuningStrategy
    resolution::Union{Int,Vector{<:Pair{<:ParameterName,Int}}}
    acceleration::AbstractResource
end

# Constructor with keywords
Grid(; resolution=10, acceleration::AbstractResource=DEFAULT_RESOURCE[]) =
    Grid(resolution, acceleration)

MLJBase.show_as_constructed(::Type{<:Grid}) = true

"""
$TYPEDEF

Container for a deterministic tuning strategy.
"""
mutable struct DeterministicTunedModel{T,M<:Deterministic} <: MLJ.Deterministic
    model::M
    tuning::T  # tuning strategy
    resampling # resampling strategy
    measure
    weights::Union{Nothing,Vector{<:Real}}
    operation
    ranges::Union{Vector{<:ParamRange},ParamRange}
    full_report::Bool
    train_best::Bool
end

"""
$TYPEDEF

Container for a probabilistic tuning strategy.
"""
mutable struct ProbabilisticTunedModel{T,M<:Probabilistic} <: MLJ.Probabilistic
    model::M
    tuning::T  # tuning strategy
    resampling # resampling strategy
    measure
    weights::Union{Nothing,AbstractVector{<:Real}}
    operation
    ranges::Union{Vector{<:ParamRange},ParamRange}
    full_report::Bool
    train_best::Bool
end

const EitherTunedModel{T,M} =
    Union{DeterministicTunedModel{T,M},ProbabilisticTunedModel{T,M}}

MLJBase.is_wrapper(::Type{<:EitherTunedModel}) = true

"""
    tuned_model = TunedModel(; model=nothing,
                             tuning=Grid(),
                             resampling=Holdout(),
                             measure=nothing,
                             weights=nothing,
                             operation=predict,
                             ranges=ParamRange[],
                             full_report=true,
                             train_best=true)

Construct a model wrapper for hyperparameter optimization of a
supervised learner.

Calling `fit!(mach)` on a machine `mach=machine(tuned_model, X, y)` or
`mach=machine(tuned_model, X, y, w)` will:

- Instigate a search, over clones of `model`, with the hyperparameter
  mutations specified by `ranges`, for a model optimizing the
  specified `measure`, using performance evaluations carried out using
  the specified `tuning` strategy and `resampling` strategy. If
  `measure` supports weights (`supports_weights(measure) == true`)
  then any `weights` specified will be passed to the measure.

- Fit an internal machine, based on the optimal model
  `fitted_params(mach).best_model`, wrapping the optimal `model`
  object in *all* the provided data `X, y` (or in `task`). Calling
  `predict(mach, Xnew)` then returns predictions on `Xnew` of this
  internal machine. The final train can be supressed by setting
  `train_best=false`.

*Important.* If a custom measure `measure` is used, and the measure is
a score, rather than a loss, be sure to check that
`MLJ.orientation(measure) == :score` to ensure maximization of the
measure, rather than minimization. Override an incorrect value with
`MLJ.orientation(::typeof(measure)) = :score`.

*Important:* If `weights` are left unspecified, and `measure` supports
sample weights, then any weight vector `w` used in constructing a
corresponding tuning machine, as in `tuning_machine =
machine(tuned_model, X, y, w)` (which is then used in *training* each
model in the search) will also be passed to `measure` for evaluation.

In the case of two-parameter tuning, a Plots.jl plot of performance
estimates is returned by `plot(mach)` or `heatmap(mach)`.

Once a tuning machine `mach` has bee trained as above, one can access
the learned parameters of the best model, using
`fitted_params(mach).best_fitted_params`. Similarly, the report of
training the best model is accessed via `report(mach).best_report`.

"""
function TunedModel(;model=nothing,
                    tuning=Grid(),
                    resampling=Holdout(),
                    measures=nothing,
                    measure=measures,
                    weights=nothing,
                    operation=predict,
                    range=ParamRange[],
                    ranges=range,
                    minimize=true,
                    full_report=true,
                    train_best=true)

    !isempty(ranges) || error("You need to specify ranges=... ")
    model !== nothing || error("You need to specify model=... ")
    model isa Supervised || error("model must be a SupervisedModel. ")

    message = clean!(model)
    isempty(message) || @info message

    if model isa Deterministic
        return DeterministicTunedModel(model, tuning, resampling,
           measure, weights, operation, ranges, full_report, train_best)
    elseif model isa Probabilistic
        return ProbabilisticTunedModel(model, tuning, resampling,
           measure, weights, operation, ranges, full_report, train_best)
    end
    error("$model does not appear to be a Supervised model.")
end

function MLJBase.clean!(model::EitherTunedModel)
    message = ""
    if model.measure === nothing
        model.measure = default_measure(model)
        message *= "No measure specified. Setting measure=$(model.measure). "
    end
    return message
end


## GRID SEARCH

function MLJBase.fit(tuned_model::EitherTunedModel{Grid,M},
                     verbosity::Integer, args...) where M

    if tuned_model.ranges isa AbstractVector
        ranges = tuned_model.ranges
    else
        ranges = [tuned_model.ranges,]
    end

    ranges isa AbstractVector{<:ParamRange} ||
        error("ranges must be a ParamRange object or a vector of " *
              "ParamRange objects. ")

    # Build a vector of resolutions, one element per range. In case of
    # OrdinalRange provide a dummy value of 5. In case of a dictionary
    # with missing keys for the NumericRange`s, use fallback of 5.
    resolution = tuned_model.tuning.resolution
    if resolution isa Vector
        val_given_field = Dict(resolution...)
        fields = keys(val_given_field)
        resolutions = map(ranges) do range
            if range.field in fields
                return val_given_field[range.field]
            else
                if range isa MLJ.NumericRange && verbosity > 0
                    @warn "No resolution specified for "*
                    "$(range.field). Will use a value of 5. "
                end
                return 5
            end
        end
    else
        resolutions = fill(resolution, length(ranges))
    end

    if tuned_model.measure isa AbstractVector
        measure = tuned_model.measure[1]
        verbosity >=0 &&
            @warn "Provided `meausure` is a vector. Using first element only. "
    else
        measure = tuned_model.measure
    end

    minimize = ifelse(orientation(measure) == :loss, true, false)

    if verbosity > 0 && tuned_model.train_best
        if minimize
            @info "Mimimizing $measure. "
        else
            @info "Maximizing $measure. "
        end
    end

    parameter_names = [string(r.field) for r in ranges]
    scales = [scale(r) for r in ranges]

    # the mutating model:
    clone = deepcopy(tuned_model.model)

    resampler = Resampler(model=clone,
                          resampling=tuned_model.resampling,
                          measure=measure,
                          weights=tuned_model.weights,
                          operation=tuned_model.operation)

    resampling_machine = machine(resampler, args...)

    # tuple of iterators over hyper-parameter values:
    iterators = map(eachindex(ranges)) do j
        range = ranges[j]
        if range isa MLJ.NominalRange
            MLJ.iterator(range)
        elseif range isa MLJ.NumericRange
            MLJ.iterator(range, resolutions[j])
        else
            throw(TypeError(:iterator, "", MLJ.ParamRange, range))
        end
    end

#    nested_iterators = copy(tuned_model.ranges, iterators)

    n_iterators = length(iterators) # same as number of ranges
    A = MLJ.unwind(iterators...)
    N = size(A, 1)

    if tuned_model.full_report
        measurements = Vector{Float64}(undef, N)
    end

    # initialize search for best model:
    best_model = deepcopy(tuned_model.model)
    best_measurement = ifelse(minimize, Inf, -Inf)
    s = ifelse(minimize, 1, -1)

    # evaluate all the models using specified resampling:
    # TODO: parallelize!

    meter = Progress(N+1, dt=0, desc="Iterating over a $N-point grid: ",
                     barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
    verbosity != 1 || next!(meter)

    for i in 1:N
        verbosity != 1 || next!(meter)

        A_row = Tuple(A[i,:])

 #       new_params = copy(nested_iterators, A_row)

        # mutate `clone` (the model to which `resampler` points):
        for k in 1:n_iterators
            field = ranges[k].field
            recursive_setproperty!(clone, field, A_row[k])
        end

        if verbosity == 2
            fit!(resampling_machine, verbosity=0)
        else
            fit!(resampling_machine, verbosity=verbosity-1)
        end
        e = evaluate(resampling_machine).measurement[1]

        if verbosity > 1
            text = prod("$(parameter_names[j])=$(A_row[j]) \t" for j in 1:length(A_row))
            text *= "measurement=$e"
            println(text)
        end

        if s*(best_measurement - e) > 0
            best_model = deepcopy(clone)
            best_measurement = e
        end

        if tuned_model.full_report
 #           models[i] = deepcopy(clone)
            measurements[i] = e
        end

    end

    fitresult = machine(best_model, args...)
    if tuned_model.train_best
        verbosity < 1 || @info "Training best model on all supplied data."

        # train best model on all the data:
        # TODO: maybe avoid using machines here and use model fit/predict?
        fit!(fitresult, verbosity=verbosity-1)
        best_report = fitresult.report
    else
        verbosity < 1 || @info "Training of best model suppressed.\n "*
        "To train tuning machine `mach` on all supplied data, call "*
        "`fit!(mach.fitresult)`."
        fitresult = tuned_model.model
        best_report = missing
    end

    pre_report = (parameter_names= permutedims(parameter_names), # row vector
                  parameter_scales=permutedims(scales),   # row vector
                  best_measurement=best_measurement,
                  best_report=best_report)

    if tuned_model.full_report
        report = merge(pre_report,
                       (parameter_values=A,
                        measurements=measurements,))
    else
        report = merge(pre_report,
                       (parameter_values=missing,
                        measurements=missing,))
    end

    cache = nothing

    return fitresult, cache, report

end

function MLJBase.fitted_params(tuned_model::EitherTunedModel, fitresult)
    if tuned_model.train_best
        return (best_model=fitresult.model,
                best_fitted_params=fitted_params(fitresult))
    else
        return (best_model=fitresult.model,
                best_fitted_params=missing)
    end
end

MLJBase.predict(tuned_model::EitherTunedModel, fitresult, Xnew) = predict(fitresult, Xnew)
MLJBase.best(model::EitherTunedModel, fitresult) = fitresult.model


## METADATA

MLJBase.supports_weights(::Type{<:EitherTunedModel{<:Any,M}}) where M =
    MLJBase.supports_weights(M)

MLJBase.load_path(::Type{<:DeterministicTunedModel}) =
    "MLJ.DeterministicTunedModel"
MLJBase.package_name(::Type{<:DeterministicTunedModel}) = "MLJ"
MLJBase.package_uuid(::Type{<:DeterministicTunedModel}) = ""
MLJBase.package_url(::Type{<:DeterministicTunedModel}) =
    "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:DeterministicTunedModel{T,M}}) where {T,M} =
    MLJBase.is_pure_julia(M)
MLJBase.input_scitype(::Type{<:DeterministicTunedModel{T,M}}) where {T,M} =
    MLJBase.input_scitype(M)
MLJBase.target_scitype(::Type{<:DeterministicTunedModel{T,M}}) where {T,M} =
    MLJBase.target_scitype(M)

MLJBase.load_path(::Type{<:ProbabilisticTunedModel}) =
    "MLJ.ProbabilisticTunedModel"
MLJBase.package_name(::Type{<:ProbabilisticTunedModel}) = "MLJ"
MLJBase.package_uuid(::Type{<:ProbabilisticTunedModel}) = ""
MLJBase.package_url(::Type{<:ProbabilisticTunedModel}) =
    "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} =
    MLJBase.is_pure_julia(M)
MLJBase.input_scitype(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} =
    MLJBase.input_scitype(M)
MLJBase.target_scitype(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} =
    MLJBase.target_scitype(M)


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

For `n` not equal to 1, multiple curves are computed, and the value of
`curve.measurements` is an array, one column for each run. This is
useful in the case of models with indeterminate fit-results, such as a
random forest.

````julia
using CSV
X, y = datanow()
atom = RidgeRegressor()
ensemble = EnsembleModel(atom=atom)
mach = machine(ensemble, X, y)
r_lambda = range(ensemble, :(atom.lambda), lower=0.1, upper=100, scale=:log10)
curve = MLJ.learning_curve!(mach; range=r_lambda)
using Plots
plot(curve.parameter_values, curve.measurements, xlab=curve.parameter_name, xscale=curve.parameter_scale)
````

If using a `Holdout` `resampling` strategy, and the specified
hyperparameter is the number of iterations in some iterative model
(and that model has an appropriately overloaded `MLJBase.update`
method) then training is not restarted from scratch for each increment
of the parameter, ie the model is trained progressively.

````julia
atom.lambda=1.0
r_n = range(ensemble, :n, lower=2, upper=150)
curves = MLJ.learning_curve!(mach; range=r_n, verbosity=3, n=5)
plot(curves.parameter_values, curves.measurements, xlab=curves.parameter_name)
````

"""
function learning_curve!(mach::Machine{<:Supervised};
                         resolution=30, resampling=Holdout(),
                         measure=rms, operation=predict,
                         range=nothing, verbosity=1, n=1)

    range !== nothing || error("No param range specified. Use range=... ")

    tuned_model = TunedModel(model=mach.model, ranges=range,
                             tuning=Grid(resolution=resolution),
                             resampling=resampling, measure=measure,
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
