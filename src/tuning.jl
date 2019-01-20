abstract type TuningStrategy <: MLJ.MLJType end

mutable struct Grid <: TuningStrategy
    resolution::Int
    parallel::Bool
end

Grid(;resolution=10, parallel=true) = Grid(resolution, parallel)

# TODO: make fitresult type `Machine` more concrete.

mutable struct TunedModel{T,M<:MLJ.Model} <: MLJ.Supervised{MLJ.Machine}
    model::M
    tuning::T
    resampling
    measure
    operation
    param_ranges::Params
    report_measurements::Bool
end

function TunedModel(;model=ConstantRegressor(),
                    tuning=Grid(),
                    resampling=Holdout(),
                    measure=rms,
                    operation=predict,
                    param_ranges=Params(),
                    report_measurements=true)
    !isempty(param_ranges) || error("No param_ranges specified.")
    return TunedModel(model, tuning, resampling,
                      measure, operation, param_ranges, report_measurements)
end

function MLJBase.fit(tuned_model::TunedModel{Grid,M}, verbosity::Int, X, y) where M

    # the mutating model:
    clone = deepcopy(tuned_model.model)
    
    resampler = Resampler(model=clone,
                          resampling=tuned_model.resampling,
                          measure=tuned_model.measure,
                          operation=tuned_model.operation)

    resampling_machine = machine(resampler, X, y)

    # tuple of `ParamRange` objects:
    ranges = MLJ.flat_values(tuned_model.param_ranges)

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
    param_iterators = copy(tuned_model.param_ranges, iterators)

    iterators = flat_values(param_iterators)
    A = unwind(iterators...)
    N = size(A, 1)

    if tuned_model.report_measurements
        models = Vector{M}(undef, N)
        measurements = Vector{Float64}(undef, N)
    end

    # initialize search for best model:
    m = 1 # model counter
    best_model = tuned_model.model
    best_measurement = Inf

    # evaluate all the models using specified resampling:
    # TODO: parallelize!

    meter = Progress(N, dt=0.5, desc="Searching a $N-point grid for best model: ",
                     barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    for i in 1:N

        verbosity < 1 || next!(meter)

        new_params = copy(param_iterators, Tuple(A[i,:]))   

        # mutate `clone` (the model to which `resampler` points):
        set_params!(clone, new_params)

        fit!(resampling_machine, verbosity=verbosity-1)
        e = evaluate(resampling_machine)
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
    else
        report = nothing
    end
    
    cache = nothing
    
    return fitresult, cache, report
    
end

MLJBase.predict(tuned_model::TunedModel, fitresult, Xnew) = predict(fitresult, Xnew)
MLJBase.best(model::TunedModel, fitresult) = fitresult.model
    
