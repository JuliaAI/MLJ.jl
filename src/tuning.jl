abstract type TuningStrategy <: MLJ.MLJType end

mutable struct Grid <: TuningStrategy
    resolution::Int
    parallel::Bool
end

Grid(;resolution=10, parallel=true) = Grid(resolution, parallel)

# TODO: make fitresult type `Machine` more concrete.

mutable struct TunedModel{T,M<:MLJ.Model} <: MLJ.Supervised{MLJ.Machine}
    model::M
    tuning_strategy::T
    resampling_strategy
    measure
    operation
    nested_ranges::Params
    report_measurements::Bool
end

function TunedModel(;model=ConstantRegressor(),
                    tuning_strategy=Grid(),
                    resampling_strategy=Holdout(),
                    measure=rms,
                    operation=predict,
                    nested_ranges=Params(),
                    report_measurements=true)
    !isempty(nested_ranges) || error("No nested_ranges specified.")
    return TunedModel(model, tuning_strategy, resampling_strategy,
                      measure, operation, nested_ranges, report_measurements)
end

function MLJBase.fit(tuned_model::TunedModel{Grid,M}, verbosity::Int, X, y) where M

    # the mutating model:
    clone = deepcopy(tuned_model.model)
    
    resampler = Resampler(model=clone,
                          resampling_strategy=tuned_model.resampling_strategy,
                          measure=tuned_model.measure,
                          operation=tuned_model.operation)

    resampling_machine = machine(resampler, X, y)

    # tuple of `ParamRange` objects:
    ranges = MLJ.flat_values(tuned_model.nested_ranges)

    # tuple of iterators over hyper-parameter values:
    iterators = map(ranges) do range
        if range isa MLJ.NominalRange
            MLJ.iterator(range)
        elseif range isa MLJ.NumericRange
            MLJ.iterator(range, tuned_model.tuning_strategy.resolution)
        else
            throw(TypeError(:iterator, "", MLJ.ParamRange, rrange))            
        end
    end

    # nested sequence of `:hyperparameter => iterator` pairs:
    param_iterators = copy(tuned_model.nested_ranges, iterators)

    iterators = flat_values(param_iterators)
    n_iterators = length(iterators)
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
                     barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)

    for i in 1:N

        verbosity < 1 || next!(meter)

        new_params = copy(param_iterators, Tuple(A[i,:]))   

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
    else
        report = nothing
    end
    
    cache = nothing
    
    return fitresult, cache, report
    
end

MLJBase.predict(tuned_model::TunedModel, fitresult, Xnew) = predict(fitresult, Xnew)
MLJBase.best(model::TunedModel, fitresult) = fitresult.model

"""
    learning_curve(model, X, ys...; resolution=30, resampling_strategy=Holdout(), measure=rms, operation=pr, param_range=nothing)

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
                        resampling_strategy=Holdout(),
                        measure=rms, operation=predict, nested_range=nothing)

    model != nothing || error("No model specified. Use learning_curve(model=..., param_range=...,)")
    nested_range != nothing || error("No param range specified. Use learning_curve(model=..., param_range=...,)")

    tuned_model = TunedModel(model=model, nested_ranges=nested_range,
                             tuning_strategy=Grid(resolution=resolution),
                             resampling_strategy=resampling_strategy, measure=measure, report_measurements=true)
    tuned = machine(tuned_model, X, ys...)
    fit!(tuned, verbosity=0)
    return tuned.report[:curve]
end


 
