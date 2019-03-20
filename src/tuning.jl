abstract type TuningStrategy <: MLJ.MLJType end

mutable struct Grid <: TuningStrategy
    resolution::Int
    parallel::Bool
end

Grid(;resolution=10, parallel=true) = Grid(resolution, parallel)

# TODO: make fitresult type `Machine` more concrete.

mutable struct DeterministicTunedModel{T,M<:Deterministic} <: MLJ.Deterministic{MLJ.Machine}
    model::M
    tuning::T  # tuning strategy 
    resampling # resampling strategy
    measure
    operation
    nested_ranges::NamedTuple
    full_report::Bool
end

mutable struct ProbabilisticTunedModel{T,M<:Probabilistic} <: MLJ.Probabilistic{MLJ.Machine}
    model::M
    tuning::T  # tuning strategy 
    resampling # resampling strategy
    measure
    operation
    nested_ranges::NamedTuple
    full_report::Bool
end

const EitherTunedModel{T,M} = Union{DeterministicTunedModel{T,M},ProbabilisticTunedModel{T,M}}

MLJBase.is_wrapper(::Type{<:EitherTunedModel}) = true

function TunedModel(;model=nothing,
                    tuning=Grid(),
                    resampling=Holdout(),
                    measure=nothing,
                    operation=predict,
                    nested_ranges=NamedTuple(),
                    full_report=true)
    
    !isempty(nested_ranges) || error("You need to specify nested_ranges=... ")
    model != nothing || error("You need to specify model=... ")

    message = clean!(model)
    isempty(message) || @info message
    
    if model isa Deterministic
        return DeterministicTunedModel(model, tuning, resampling,
                      measure, operation, nested_ranges, full_report)
    elseif model isa Probabilistic
        return ProbabilisticTunedModel(model, tuning, resampling,
                      measure, operation, nested_ranges, full_report)
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
    best_measurement = Inf

    # evaluate all the models using specified resampling:
    # TODO: parallelize!

    meter = Progress(N+1, dt=0, desc="Iterating over an $N-point grid: ",
                     barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
    verbosity < 1 || next!(meter)
    for i in 1:N

        verbosity < 1 || next!(meter)

        new_params = copy(nested_iterators, Tuple(A[i,:]))   

        # mutate `clone` (the model to which `resampler` points):
        set_params!(clone, new_params)

        fit!(resampling_machine, verbosity=verbosity-1)
        e = mean(evaluate(resampling_machine))
        if e < best_measurement
            best_model = deepcopy(clone)
            best_measurement = e
        end

        if tuned_model.full_report
 #           models[i] = deepcopy(clone)
            measurements[i] = e
        end
        
    end

    verbosity < 1 || @info "Training best model on all supplied data."

    # train best model on all the data:
    # TODO: maybe avoid using machines here and use model fit/predict?
    fitresult = machine(best_model, X, y)
    fit!(fitresult, verbosity=verbosity-1)

    scales=scale.(flat_values(tuned_model.nested_ranges)) |> collect

    if tuned_model.full_report
        report = (# models=models,
                  # best_model=best_model,
                  parameter_names= permutedims(flat_keys(tuned_model.nested_ranges)), # row vector
                  parameter_scales=permutedims(scales),  # row vector
                  parameter_values=A,
                  measurements=measurements,
                  best_measurement=best_measurement)
    else
        report = (# models=[deepcopy(clone),][1:0],         # empty vector
                  # best_model=best_model,
                  parameter_names= permutedims(flat_keys(tuned_model.nested_ranges)), # row vector
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
MLJBase.input_scitypes(::Type{<:DeterministicTunedModel{T,M}}) where {T,M} = MLJBase.input_scitypes(M)
MLJBase.input_is_multivariate(::Type{<:DeterministicTunedModel{T,M}}) where {T,M} = MLJBase.input_is_multivariate(M)
MLJBase.target_scitype(::Type{<:DeterministicTunedModel{T,M}}) where {T,M} = MLJBase.target_scitype(M)

MLJBase.load_path(::Type{<:ProbabilisticTunedModel}) = "MLJ.ProbabilisticTunedModel"
MLJBase.package_name(::Type{<:ProbabilisticTunedModel}) = "MLJ"
MLJBase.package_uuid(::Type{<:ProbabilisticTunedModel}) = ""
MLJBase.package_url(::Type{<:ProbabilisticTunedModel}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} = MLJBase.is_pure_julia(M)
MLJBase.input_scitypes(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} = MLJBase.input_scitypes(M)
MLJBase.target_scitype(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} = MLJBase.target_scitype(M)
MLJBase.input_is_multivariate(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} = MLJBase.input_is_multivariate(M)


## LEARNING CURVES 

"""
    curve = learning_curve!(mach; resolution=30, resampling=Holdout(), measure=rms, operation=predict, nested_range=nothing)

Given a supervised machine `mach`, returns a named tuple of objects
needed to generate a plot of performance measurements, as a function
of the single hyperparameter specified in `nested_range`. The tuple `curve`
has the following keys: `:parameter_name`, `:parameter_scale`,
`:parameter_values`, `:measurements`.

````julia
X, y = datanow()
atom = RidgeRegressor()
ensemble = EnsembleModel(atom=atom)
mach = machine(ensemble, X, y)
r_lambda = range(atom, :lambda, lower=0.1, upper=100, scale=:log10)
curve = MLJ.learning_curve!(mach; nested_range=(atom=(lambda=r_lambda,),))
using Plots
plot(curve.parameter_values, curve.measurements, xscale=curve.parameter_scale)
````

Smart fitting applies. For example, if the model is an ensemble model,
and the hyperparemeter parameter is `n`, then atomic models are
progressively added to the ensemble, not recomputed from scratch for
each new value of `n`.

````julia
atom.lambda=1.0
r_n = range(ensemble, :n, lower=2, upper=100)
curve2 = MLJ.learning_curve!(mach; nested_range=(n=r_n,), verbosity=3)
plot(curve2.parameter_values, curve2.measurements)
````

"""
function learning_curve!(mach::Machine{<:Supervised};
                        resolution=30,
                        resampling=Holdout(),
                        measure=rms, operation=predict, nested_range=nothing, verbosity=0)

    nested_range != nothing || error("No param range specified. Use nested_range=... ")

    tuned_model = TunedModel(model=mach.model, nested_ranges=nested_range,
                             tuning=Grid(resolution=resolution),
                             resampling=resampling, measure=measure, full_report=true)
    tuned = machine(tuned_model, mach.args...)
    fit!(tuned, verbosity=verbosity)
    report = tuned.report
    return (parameter_name=report.parameter_names[1],
            parameter_scale=report.parameter_scales[1],
            parameter_values=[report.parameter_values[:, 1]...],
            measurements=report.measurements)
end


 
