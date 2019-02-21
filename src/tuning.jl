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
    nested_ranges::Params
    report_measurements::Bool
end

mutable struct ProbabilisticTunedModel{T,M<:Probabilistic} <: MLJ.Probabilistic{MLJ.Machine}
    model::M
    tuning::T  # tuning strategy 
    resampling # resampling strategy
    measure
    operation
    nested_ranges::Params
    report_measurements::Bool
end

const EitherTunedModel{T,M} = Union{DeterministicTunedModel{T,M},ProbabilisticTunedModel{T,M}}

function TunedModel(;model=nothing,
                    tuning=Grid(),
                    resampling=Holdout(),
                    measure=nothing,
                    operation=predict,
                    nested_ranges=Params(),
                    report_measurements=true)
    
    !isempty(nested_ranges) || error("You need to specify nested_ranges=... ")
    model != nothing || error("You need to specify model=... ")

    message = clean!(model)
    isempty(message) || @info message
    
    if model isa Deterministic
        return DeterministicTunedModel(model, tuning, resampling,
                      measure, operation, nested_ranges, report_measurements)
    elseif model isa Probabilistic
        return ProbabilisticTunedModel(model, tuning, resampling,
                      measure, operation, nested_ranges, report_measurements)
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

    # nested sequence of `:hyperparameter => iterator` pairs:
    nested_iterators = copy(tuned_model.nested_ranges, iterators)

    # iterators = MLJ.flat_values(nested_iterators)
    
    n_iterators = length(iterators)
    A = MLJ.unwind(iterators...)
    N = size(A, 1)

    if tuned_model.report_measurements
        models = Vector{M}(undef, N)
        measurements = Vector{Float64}(undef, N)
    end

    # initialize search for best model:
    best_model = deepcopy(tuned_model.model)
    best_measurement = Inf

    # evaluate all the models using specified resampling:
    # TODO: parallelize!

    meter = Progress(N+1, dt=0, desc="Searching a $N-point grid for best model: ",
                     barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
    next!(meter)
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

        if tuned_model.report_measurements
            models[i] = deepcopy(clone)
            measurements[i] = e
        end
        
    end

    verbosity < 1 || @info "Training best model on all supplied data."

    # train best model on all the data:
    # TODO: maybe avoid using machines here and use model fit/predict?
    fitresult = machine(best_model, X, y)
    fit!(fitresult, verbosity=verbosity-1)

    if tuned_model.report_measurements
        report = Dict{Symbol, Any}()
        report[:models] = models
        report[:measurements] = measurements
        report[:best_model] = best_model
        report[:best_measurement] = best_measurement
        if n_iterators == 1
            report[:curve] = ([A[:,1]...], measurements)
        end
        # return parameter names as row vector to correspond to layout of values:
        report[:parameter_names] = reshape(flat_keys(tuned_model.nested_ranges), 1, :)
        report[:parameter_values] = A
    else
        report = nothing
    end
    
    cache = nothing
    
    return fitresult, cache, report
    
end

MLJBase.predict(tuned_model::EitherTunedModel, fitresult, Xnew) = predict(fitresult, Xnew)
MLJBase.best(model::EitherTunedModel, fitresult) = fitresult.model


## METADATA

MLJBase.load_path(::Type{<:DeterministicTunedModel}) = "MLJ.DeterministicTunedModel"
MLJBase.package_name(::Type{<:DeterministicTunedModel}) = "MLJ"
MLJBase.package_uuid(::Type{<:DeterministicTunedModel}) = ""
MLJBase.package_url(::Type{<:DeterministicTunedModel}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:DeterministicTunedModel{T,M}}) where {T,M} = MLJBase.is_pure_julia(M)
MLJBase.input_kinds(::Type{<:DeterministicTunedModel{T,M}}) where {T,M} = MLJBase.input_kinds(M)
MLJBase.output_kind(::Type{<:DeterministicTunedModel{T,M}}) where {T,M} = MLJBase.output_kind(M)
MLJBase.output_quantity(::Type{<:DeterministicTunedModel{T,M}}) where {T,M} = MLJBase.output_quantity(M)

MLJBase.load_path(::Type{<:ProbabilisticTunedModel}) = "MLJ.ProbabilisticTunedModel"
MLJBase.package_name(::Type{<:ProbabilisticTunedModel}) = "MLJ"
MLJBase.package_uuid(::Type{<:ProbabilisticTunedModel}) = ""
MLJBase.package_url(::Type{<:ProbabilisticTunedModel}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} = MLJBase.is_pure_julia(M)
MLJBase.input_kinds(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} = MLJBase.input_kinds(M)
MLJBase.output_kind(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} = MLJBase.output_kind(M)
MLJBase.output_quantity(::Type{<:ProbabilisticTunedModel{T,M}}) where {T,M} = MLJBase.output_quantity(M)


## LEARNING CURVES (1D TUNING)

"""
    learning_curve(model, X, ys...; resolution=30, resampling=Holdout(), measure=rms, operation=pr, param_range=nothing)

Returns `(u, v)` where `u` is a vector of hyperparameter values, and
`v` the corresponding performance estimates. 

````julia
X, y = datanow()
atom = RidgeRegressor()
model = EnsembleModel(atom=atom)
r = range(atom, :lambda, lower=0.1, upper=100, scale=:log10)
param_range = Params(:atom => Params(:lambda => r))
u, v = MLJ.learning_curve(model, X, y; param_range = param_range) 
````

"""
function learning_curve(model::Supervised, X, ys...;
                        resolution=30,
                        resampling=Holdout(),
                        measure=rms, operation=predict, nested_range=nothing)

    model != nothing || error("No model specified. Use learning_curve(model=..., param_range=...,)")
    nested_range != nothing || error("No param range specified. Use learning_curve(model=..., param_range=...,)")

    tuned_model = TunedModel(model=model, nested_ranges=nested_range,
                             tuning=Grid(resolution=resolution),
                             resampling=resampling, measure=measure, report_measurements=true)
    tuned = machine(tuned_model, X, ys...)
    fit!(tuned, verbosity=0)
    return tuned.report[:curve]
end


 
