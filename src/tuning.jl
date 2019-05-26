abstract type TuningStrategy <: MLJ.MLJType end

mutable struct Grid <: TuningStrategy
    resolution::Int
    parallel::Bool
end

Grid(;resolution=10, parallel=true) = Grid(resolution, parallel)
MLJBase.show_as_constructed(::Type{<:Grid}) = true

# TODO: make fitresult type `Machine` more concrete.

mutable struct DeterministicTunedModel{T,M<:Deterministic} <: MLJ.Deterministic
    model::M
    tuning::T  # tuning strategy 
    resampling # resampling strategy
    measure
    operation
    nested_ranges::NamedTuple
    minimize::Bool
    full_report::Bool
    train_best::Bool 
end

mutable struct ProbabilisticTunedModel{T,M<:Probabilistic} <: MLJ.Probabilistic
    model::M
    tuning::T  # tuning strategy 
    resampling # resampling strategy
    measure
    operation
    nested_ranges::NamedTuple
    minimize::Bool
    full_report::Bool
    train_best::Bool
end

const EitherTunedModel{T,M} = Union{DeterministicTunedModel{T,M},ProbabilisticTunedModel{T,M}}

MLJBase.is_wrapper(::Type{<:EitherTunedModel}) = true

"""
    tuned_model = TunedModel(; model=nothing,
                             tuning=Grid(),
                             resampling=Holdout(),
                             measure=nothing,
                             operation=predict,
                             nested_ranges=NamedTuple(),
                             minimize=true,
                             full_report=true)

Construct a model wrapper for hyperparameter optimization of a
supervised learner.

Calling `fit!(mach)` on a machine `mach=machine(tuned_model, X, y)`
will: (i) Instigate a search, over clones of `model` with the
hyperparameter mutations specified by `nested_ranges`, for that model
optimizing the specified `measure`, according to evaluations carried
out using the specified `tuning` strategy and `resampling` strategy;
and (ii) Fit a machine, `mach_optimal = mach.fitresult`, wrapping the
optimal `model` object in *all* the provided data `X, y`. Calling
`predict(mach, Xnew)` then returns predictions on `Xnew` of the
machine `mach_optimal`.

If `measure` is a score, rather than a loss, specify `minimize=false`.

The optimal clone of `model` is accessible as
`fitted_params(mach).best_model`. In the case of two-parameter tuning,
a Plots.jl plot of performance estimates is returned by `plot(mach)`
or `heatmap(mach)`.

"""
function TunedModel(;model=nothing,
                    tuning=Grid(),
                    resampling=Holdout(),
                    measure=nothing,
                    operation=predict,
                    nested_ranges=NamedTuple(),
                    minimize=true,
                    full_report=true,
                    train_best=true)
    
    !isempty(nested_ranges) || error("You need to specify nested_ranges=... ")
    model != nothing || error("You need to specify model=... ")

    message = clean!(model)
    isempty(message) || @info message
    
    if model isa Deterministic
        return DeterministicTunedModel(model, tuning, resampling,
           measure, operation, nested_ranges, minimize, full_report, train_best)
    elseif model isa Probabilistic
        return ProbabilisticTunedModel(model, tuning, resampling,
           measure, operation, nested_ranges, minimize, full_report, train_best)
    end
    error("$model does not appear to be a Supervised model.")
end

function MLJBase.clean!(model::EitherTunedModel)
    message = ""
    if model.measure == nothing
        model.measure = default_measure(model)
        message *= "No measure specified. Using measure=$(model.measure). "
    end
    return message
end

function MLJBase.fit(tuned_model::EitherTunedModel{Grid,M}, verbosity::Int, X, y) where M

    parameter_names = flat_keys(tuned_model.nested_ranges)
    
    # the mutating model:
    clone = deepcopy(tuned_model.model)

    measure = tuned_model.measure

    resampler = Resampler(model=clone,
                          resampling=tuned_model.resampling,
                          measure=measure,
                          operation=tuned_model.operation)

    resampling_machine = machine(resampler, X, y)

    # tuple of `ParamRange` objects:
    ranges = MLJ.flat_values(tuned_model.nested_ranges)

    # tuple of iterators over hyper-parameter values:
    iterators = map(ranges) do range
        if range isa MLJ.NominalRange
            MLJ.iterator(range)
        elseif range isa MLJ.NumericRange
            MLJ.iterator(range, tuned_model.tuning.resolution)
        else
            throw(TypeError(:iterator, "", MLJ.ParamRange, rrange))            
        end
    end

    nested_iterators = copy(tuned_model.nested_ranges, iterators)

    n_iterators = length(iterators)
    A = MLJ.unwind(iterators...)
    N = size(A, 1)

    if tuned_model.full_report
#        models = Vector{M}(undef, N)
        measurements = Vector{Float64}(undef, N)
    end

    # initialize search for best model:
    best_model = deepcopy(tuned_model.model)
    best_measurement =
        tuned_model.minimize ? Inf : -Inf
    s =
        tuned_model.minimize ? 1 : -1

    # evaluate all the models using specified resampling:
    # TODO: parallelize!

    meter = Progress(N+1, dt=0, desc="Iterating over a $N-point grid: ",
                     barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
    verbosity != 1 || next!(meter)
    for i in 1:N

        verbosity != 1 || next!(meter)

        A_row = Tuple(A[i,:])

        new_params = copy(nested_iterators, A_row)   

        # mutate `clone` (the model to which `resampler` points):
        set_params!(clone, new_params)

        if verbosity == 2
            fit!(resampling_machine, verbosity=0)

        else
            fit!(resampling_machine, verbosity=verbosity-1)
        end
        e = mean(evaluate(resampling_machine))

        if verbosity > 1
            text = reduce(*, ["$(parameter_names[j])=$(A_row[j]) \t"
                              for j in 1:length(A_row)])
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

    if tuned_model.train_best
        verbosity < 1 || @info "Training best model on all supplied data."

        # train best model on all the data:
        # TODO: maybe avoid using machines here and use model fit/predict?
        fitresult = machine(best_model, X, y)
        fit!(fitresult, verbosity=verbosity-1)
    else
        fitresult = tuned_model.model
    end

    scales=scale.(flat_values(tuned_model.nested_ranges)) |> collect

    if tuned_model.full_report
        report = (# models=models,
                  # best_model=best_model,
                  parameter_names= permutedims(parameter_names), # row vector
                  parameter_scales=permutedims(scales),  # row vector
                  parameter_values=A,
                  measurements=measurements,
                  best_measurement=best_measurement)
    else
        report = (# models=[deepcopy(clone),][1:0],         # empty vector
                  # best_model=best_model,
                  parameter_names= permutedims(parameter_names), # row vector
                  parameter_scales=permutedims(scales),   # row vector
                  parameter_values=A[1:0,1:0],            # empty matrix
                  measurements=[best_measurement, ][1:0], # empty vector
                  best_measurement=best_measurement)
    end
    
    cache = nothing
    
    return fitresult, cache, report
    
end

MLJBase.fitted_params(::EitherTunedModel, fitresult) = (best_model=fitresult.model,)

MLJBase.predict(tuned_model::EitherTunedModel, fitresult, Xnew) = predict(fitresult, Xnew)
MLJBase.best(model::EitherTunedModel, fitresult) = fitresult.model


## METADATA

MLJBase.load_path(::Type{<:DeterministicTunedModel}) = "MLJ.DeterministicTunedModel"
MLJBase.package_name(::Type{<:DeterministicTunedModel}) = "MLJ"
MLJBase.package_uuid(::Type{<:DeterministicTunedModel}) = ""
MLJBase.package_url(::Type{<:DeterministicTunedModel}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:DeterministicTunedModel{T,M}}) where {T,M} = MLJBase.is_pure_julia(M)
MLJBase.input_scitype_union(::Type{<:DeterministicTunedModel{T,M}}) where {T,M} = MLJBase.input_scitype_union(M)
MLJBase.input_is_multivariate(::Type{<:DeterministicTunedModel{T,M}}) where {T,M} = MLJBase.input_is_multivariate(M)
MLJBase.target_scitype_union(::Type{<:DeterministicTunedModel{T,M}}) where {T,M} = MLJBase.target_scitype_union(M)

MLJBase.load_path(::Type{<:ProbabilisticTunedModel}) = "MLJ.ProbabilisticTunedModel"
MLJBase.package_name(::Type{<:ProbabilisticTunedModel}) = "MLJ"
MLJBase.package_uuid(::Type{<:ProbabilisticTunedModel}) = ""
MLJBase.package_url(::Type{<:ProbabilisticTunedModel}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} = MLJBase.is_pure_julia(M)
MLJBase.input_scitype_union(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} = MLJBase.input_scitype_union(M)
MLJBase.target_scitype_union(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} = MLJBase.target_scitype_union(M)
MLJBase.input_is_multivariate(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} = MLJBase.input_is_multivariate(M)


## LEARNING CURVES 

"""
    curve = learning_curve!(mach; resolution=30, resampling=Holdout(), measure=rms, operation=predict, nested_range=nothing, n=1)

Given a supervised machine `mach`, returns a named tuple of objects
needed to generate a plot of performance measurements, as a function
of the single hyperparameter specified in `nested_range`. The tuple `curve`
has the following keys: `:parameter_name`, `:parameter_scale`,
`:parameter_values`, `:measurements`.

For `n` not equal to 1, multiple curves are computed, and the value of
`curve.measurements` is an array, one column for each run. This is
useful in the case of models with indeterminate fit-results, such as a
random forest.

````julia
X, y = datanow()
atom = RidgeRegressor()
ensemble = EnsembleModel(atom=atom)
mach = machine(ensemble, X, y)
r_lambda = range(atom, :lambda, lower=0.1, upper=100, scale=:log10)
curve = MLJ.learning_curve!(mach; nested_range=(atom=(lambda=r_lambda,),))
using Plots
plot(curve.parameter_values, curve.measurements, xlab=curve.parameter_name, xscale=curve.parameter_scale)
````

If the specified hyperparameter is the number of iterations in some
iterative model (and that model has an appropriately overloaded
`MLJBase.update` method) then training is not restarted from scratch
for each increment of the parameter, ie the model is trained
progressively.

````julia
atom.lambda=1.0
r_n = range(ensemble, :n, lower=2, upper=150)
curves = MLJ.learning_curve!(mach; nested_range=(n=r_n,), verbosity=3, n=5)
plot(curves.parameter_values, curves.measurements, xlab=curves.parameter_name)
````

"""
function learning_curve!(mach::Machine{<:Supervised};
                        resolution=30,
                        resampling=Holdout(),
                         measure=rms, operation=predict, nested_range=nothing, verbosity=1, n=1)

    nested_range != nothing || error("No param range specified. Use nested_range=... ")

    tuned_model = TunedModel(model=mach.model, nested_ranges=nested_range,
                             tuning=Grid(resolution=resolution),
                             resampling=resampling, measure=measure,
                             full_report=true, train_best=false)
    tuned = machine(tuned_model, mach.args...)

    measurements = reduce(hcat, [(fit!(tuned, verbosity=verbosity);
                                  tuned.report.measurements) for c in 1:n])
    report = tuned.report
    parameter_name=report.parameter_names[1]
    parameter_scale=report.parameter_scales[1]
    parameter_values=[report.parameter_values[:, 1]...]
    measurements_ =
        n == 1 ? [measurements...] : measurements
    
    return (parameter_name=parameter_name,
            parameter_scale=parameter_scale,
            parameter_values=parameter_values,
            measurements = measurements_)
end


 
