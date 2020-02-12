# shorter display of MersenneTwister:
Base.show(stream::IO, t::Random.MersenneTwister) =
    print(stream, "MersenneTwister($(t.seed)) @ $(t.idxF)")


## ENSEMBLES OF FITRESULTS

# Atom is atomic model type, eg, DecisionTree
# R will be the tightest type of the atom fit-results.
mutable struct WrappedEnsemble{R,Atom <: Supervised} <: MLJType
    atom::Atom
    ensemble::Vector{R}
end

# A corner case here is wrapped ensembles of categorical elements (eg,
# ensembles of fitresults for ConstantClassifier). These appear
# because doing comprehension with categorical elements gives
# CategoricalVector instead of Vector, but Vector is required in above
# struct definition.
function WrappedEnsemble(atom, ensemble::AbstractVector{L}) where L
    ensemble_vec = Vector{L}(undef, length(ensemble))
    for k in eachindex(ensemble)
        ensemble_vec[k] = ensemble[k]
    end
    return WrappedEnsemble(atom, ensemble_vec)
end

# to enable trait-based dispatch of predict:
function predict(wens::WrappedEnsemble{R,Atom}, atomic_weights, Xnew
                 ) where {R,Atom<:Deterministic}
    predict(wens, atomic_weights, Xnew, Deterministic, target_scitype(Atom))
end

function predict(wens::WrappedEnsemble{R,Atom}, atomic_weights, Xnew
                 ) where {R,Atom<:Probabilistic}
    predict(wens, atomic_weights, Xnew, Probabilistic, target_scitype(Atom))
end

function predict(wens::WrappedEnsemble, atomic_weights, Xnew,
                 ::Type{Deterministic}, ::Type{<:AbstractVector{<:Finite}})
    # atomic_weights ignored in this case
    ensemble = wens.ensemble
    atom     = wens.atom
    n_atoms = length(ensemble)

    n_atoms > 0  || @error "Empty ensemble cannot make predictions."

    # TODO: make this more memory efficient but note that the type of
    # Xnew is unknown (ie, model dependent)
    preds_gen   = (predict(atom, fitresult, Xnew) for fitresult in ensemble)
    predictions = hcat(preds_gen...)

    classes    = levels(predictions)
    n          = size(predictions, 1)
    prediction =
        categorical(vcat([mode(predictions[i,:]) for i in 1:n], classes))[1:n]
    return prediction
end

function predict(wens::WrappedEnsemble, atomic_weights, Xnew,
                 ::Type{Deterministic}, ::Type{<:AbstractVector{<:Continuous}})
    # considering atomic weights
    ensemble = wens.ensemble
    atom     = wens.atom
    n_atoms  = length(ensemble)

    n_atoms > 0  || @error "Empty ensemble cannot make predictions."

    # TODO: make more memory efficient:
    preds_gen   = (atomic_weights[k] * predict(atom, ensemble[k], Xnew)
                    for k in 1:n_atoms)
    predictions = hcat(preds_gen...)
    prediction  = [sum(predictions[i,:]) for i in 1:size(predictions, 1)]

    return prediction
end

function predict(wens::WrappedEnsemble, atomic_weights, Xnew,
                 ::Type{Probabilistic}, ::Type{<:AbstractVector{<:Finite}})
    ensemble = wens.ensemble
    atom     = wens.atom
    n_atoms  = length(ensemble)

    n_atoms > 0  || @error "Empty ensemble cannot make predictions."

    # TODO: make this more memory efficient but note that the type of
    # Xnew is unknown (ie, model dependent):
    # a matrix of probability distributions:
    preds_gen   = (predict(atom, fitresult, Xnew) for fitresult in ensemble)
    predictions = hcat(preds_gen...)
    n_rows      = size(predictions, 1)

    # the weighted averages over the ensemble of the discrete pdf's:
    predictions = [average([predictions[i, k] for k in 1:n_atoms], weights=atomic_weights) for i in 1:n_rows]

    return predictions
end

function predict(wens::WrappedEnsemble, atomic_weights, Xnew,
                 ::Type{Probabilistic}, ::Type{<:AbstractVector{<:Continuous}})
    ensemble = wens.ensemble
    atom     = wens.atom
    n_atoms  = length(ensemble)

    n_atoms > 0  || @error "Empty ensemble cannot make predictions."

    # TODO: make this more memory efficient but note that the type of
    # Xnew is unknown (ie, model dependent):
    # a matrix of probability distributions:
    preds_gen   = (predict(atom, fitresult, Xnew) for fitresult in ensemble)
    predictions = hcat(preds_gen...)

    # n_rows = size(predictions, 1)
    # # the weighted average over the ensemble of the pdf means and pdf variances:
    # μs  = [sum([atomic_weights[k]*mean(predictions[i,k]) for k in 1:n_atoms]) for i in 1:n_rows]
    # σ2s = [sum([atomic_weights[k]*var(predictions[i,k]) for k in 1:n_atoms]) for i in 1:n_rows]

    # # a vector of normal probability distributions:
    # prediction = [Distributions.Normal(μs[i], sqrt(σ2s[i])) for i in 1:n_rows]

    prediction = [Distributions.MixtureModel(predictions[i,:], atomic_weights) for i in 1:size(predictions, 1)]

    return prediction

end


## CORE ENSEMBLE-BUILDING FUNCTIONS

# for when out-of-bag performance estimates are requested:
function get_ensemble_and_indices(atom::Supervised, verbosity, n, n_patterns,
                      n_train, rng, progress_meter, args...)

    ensemble_indices =
        [StatsBase.sample(rng, 1:n_patterns, n_train, replace=false)
         for i in 1:n]
    ensemble = map(ensemble_indices) do train_rows
        verbosity == 1 && next!(progress_meter)
        verbosity < 2 ||  print("#")
        atom_fitresult, atom_cache, atom_report =
            fit(atom, verbosity - 1, [selectrows(arg, train_rows) for arg in args]...)
        atom_fitresult
    end
    verbosity < 1 || println()

    return (ensemble, ensemble_indices)

end

# for when out-of-bag performance estimates are not requested:
function get_ensemble(atom::Supervised, verbosity, n, n_patterns,
                      n_train, rng, progress_meter, args...)

    # define generator of training rows:
    if n_train == n_patterns
        # keep deterministic by avoiding re-ordering:
        ensemble_indices = (1:n_patterns for i in 1:n)
    else
        ensemble_indices =
            (StatsBase.sample(rng, 1:n_patterns, n_train, replace=false)
             for i in 1:n)
    end

    ensemble = map(ensemble_indices) do train_rows
        verbosity == 1 && next!(progress_meter)
        verbosity < 2 ||  print("#")
        atom_fitresult, atom_cache, atom_report =
            fit(atom, verbosity - 1, [selectrows(arg, train_rows) for
                                      arg in args]...)
        atom_fitresult
    end
    verbosity < 1 || println()

    return ensemble

end


# for combining vectors:
_reducer(p, q) = vcat(p, q)
# for combining 2-tuples of vectors:
_reducer(p::Tuple, q::Tuple) = (vcat(p[1], q[1]), vcat(p[2], q[2]))



## ENSEMBLE MODEL FOR DETERMINISTIC MODELS

mutable struct DeterministicEnsembleModel{Atom<:Deterministic} <: Deterministic
    atom::Atom
    atomic_weights::Vector{Float64}
    bagging_fraction::Float64
    rng::Union{Int,AbstractRNG}
    n::Int
    acceleration::AbstractResource
    out_of_bag_measure # TODO: type this
end

function clean!(model::DeterministicEnsembleModel)

    target_scitype(model.atom) <: Union{AbstractVector{<:Finite}, AbstractVector{<:Continuous}} ||
        error("`atom` has unsupported target_scitype "*
              "`$(target_scitype(model.atom))`. ")

    message = ""

    if model.bagging_fraction > 1 || model.bagging_fraction <= 0
        message = message*"`bagging_fraction` should be "*
        "in the range (0,1]. Reset to 1. "
        model.bagging_fraction = 1.0
    end

    if target_scitype(model.atom) <: AbstractVector{<:Finite} && !isempty(model.atomic_weights)
        message = message*"atomic_weights will be ignored to form predictions. "
    elseif !isempty(model.atomic_weights)
        total = sum(model.atomic_weights)
        if !(total ≈ 1.0)
            message = message*"atomic_weights should sum to one and are being automatically normalized. "
            model.atomic_weights = model.atomic_weights/total
        end
    end

    return message

end

# constructor to infer type automatically:
DeterministicEnsembleModel(atom::Atom, atomic_weights,
                           bagging_fraction, rng, n, acceleration, out_of_bag_measure) where Atom<:Deterministic =
                               DeterministicEnsembleModel{Atom}(atom, atomic_weights,
                                                                   bagging_fraction, rng, n, acceleration, out_of_bag_measure)

# lazy keyword constructors:
function DeterministicEnsembleModel(;atom=DeterministicConstantClassifier(),
                                    atomic_weights=Float64[],
                                    bagging_fraction=0.8,
                                    rng=Random.GLOBAL_RNG,
                                    n::Int=100,
                                    acceleration=default_resource(),
                                    out_of_bag_measure=[])

    model = DeterministicEnsembleModel(atom, atomic_weights, bagging_fraction, rng,
                                       n, acceleration, out_of_bag_measure)

    message = clean!(model)
    isempty(message) || @warn message

    return model
end


## ENSEMBLE MODEL FOR PROBABILISTIC MODELS

mutable struct ProbabilisticEnsembleModel{Atom<:Probabilistic} <: Probabilistic
    atom::Atom
    atomic_weights::Vector{Float64}
    bagging_fraction::Float64
    rng::Union{Int, AbstractRNG}
    n::Int
    acceleration::AbstractResource
    out_of_bag_measure
end

function clean!(model::ProbabilisticEnsembleModel)

    message = ""

    if model.bagging_fraction > 1 || model.bagging_fraction <= 0
        message = message*"`bagging_fraction` should be "*
        "in the range (0,1]. Reset to 1. "
        model.bagging_fraction = 1.0
    end

    if !isempty(model.atomic_weights)
        total = sum(model.atomic_weights)
        if !(total ≈ 1.0)
            message = message*"atomic_weights should sum to one and are being automatically normalized. "
            model.atomic_weights = model.atomic_weights/total
        end
    end

    return message

end

# constructor to infer type automatically:
ProbabilisticEnsembleModel(atom::Atom, atomic_weights, bagging_fraction, rng, n, acceleration, out_of_bag_measure) where Atom<:Probabilistic =
                               ProbabilisticEnsembleModel{Atom}(atom, atomic_weights, bagging_fraction, rng, n, acceleration, out_of_bag_measure)

# lazy keyword constructor:
function ProbabilisticEnsembleModel(;atom=ConstantProbabilisticClassifier(),
                                    atomic_weights=Float64[],
                                    bagging_fraction=0.8,
                                    rng=Random.GLOBAL_RNG,
                                    n::Int=100,
                                    acceleration=default_resource(),
                                    out_of_bag_measure=[])

    model = ProbabilisticEnsembleModel(atom, atomic_weights, bagging_fraction, rng, n, acceleration, out_of_bag_measure)

    message = clean!(model)
    isempty(message) || @warn message

    return model
end


## COMMON CONSTRUCTOR

"""
    EnsembleModel(atom=nothing,
                  atomic_weights=Float64[],
                  bagging_fraction=0.8,
                  n=100,
                  rng=GLOBAL_RNG,
                  acceleration=default_resource(),
                  out_of_bag_measure=[])

Create a model for training an ensemble of `n` learners, with optional
bagging, each with associated model `atom`. Ensembling is useful if
`fit!(machine(atom, data...))` does not create identical models on
repeated calls (ie, is a stochastic model, such as a decision tree
with randomized node selection criteria), or if `bagging_fraction` is
set to a value less than 1.0, or both. The constructor fails if no
`atom` is specified.

Only atomic models supporting targets with scitype
`AbstractVector{<:Finite}` (univariate classifiers) or
`AbstractVector{<:Continuous}` (univariate regressors) are supported.

If `rng` is an integer, then `MersenneTwister(rng)` is the random
number generator used for bagging. Otherwise some `AbstractRNG` object
is expected.

The atomic predictions are weighted according to the vector
`atomic_weights` (to allow for external optimization) except in the
case that `atom` is a `Deterministic` classifier. Uniform
atomic weights are used if `weight` has zero length.

The ensemble model is `Deterministic` or `Probabilistic`, according to
the corresponding supertype of `atom`. In the case of deterministic
classifiers (`target_scitype(atom) <: Abstract{<:Finite}`), the
predictions are majority votes, and for regressors
(`target_scitype(atom)<: AbstractVector{<:Continuous}`) they are
ordinary averages.  Probabilistic predictions are obtained by
averaging the atomic probability distribution/mass functions; in
particular, for regressors, the ensemble prediction on each input
pattern has the type `MixtureModel{VF,VS,D}` from the Distributions.jl
package, where `D` is the type of predicted distribution for `atom`.

The `acceleration` keyword argument is used to specify the compute resource (a
subtype of `ComputationalResources.AbstractResource`) that will be used to
accelerate/parallelize ensemble fitting.

If a single measure or non-empty vector of measures is specified by
`out_of_bag_measure`, then out-of-bag estimates of performance are
written to the trainig report (call `report` on the trained
machine wrapping the ensemble model).

*Important:* If sample weights `w` (as opposed to atomic weights) are
specified when constructing a machine for the ensemble model, as in
`mach = machine(ensemble_model, X, y, w)`, then `w` is used by any
measures specified in `out_of_bag_measure` that support sample
weights.

"""
function EnsembleModel(; args...)
    d = Dict(args)
    :atom in keys(d) ||
        error("No atomic model specified. Use EnsembleModel(atom=...)")
    if d[:atom] isa Deterministic
        return DeterministicEnsembleModel(; d...)
    elseif d[:atom] isa Probabilistic
        return ProbabilisticEnsembleModel(; d...)
    end
    error("$(d[:atom]) does not appear to be a Supervised model.")
end


## THE COMMON FIT AND PREDICT METHODS

const EitherEnsembleModel{Atom} =
    Union{DeterministicEnsembleModel{Atom}, ProbabilisticEnsembleModel{Atom}}

MLJBase.is_wrapper(::Type{<:EitherEnsembleModel}) = true

function _fit(res::CPU1, func, verbosity, stuff)
    atom, n, n_patterns, n_train, rng, progress_meter, args = stuff
    verbosity < 2 ||  @info "One hash per new atom trained: "
    return func(atom, verbosity, n, n_patterns, n_train, rng,
                progress_meter, args...)
end
function _fit(res::CPUProcesses, func, verbosity, stuff)
    atom, n, n_patterns, n_train, rng, progress_meter, args = stuff
    if verbosity > 0
        println("Ensemble-building in parallel on $(nworkers()) processors.")
    end
    chunk_size = div(n, nworkers())
    left_over = mod(n, nworkers())
    return @distributed (_reducer) for i = 1:nworkers()
        if i != nworkers()
            func(atom, 0, chunk_size, n_patterns, n_train,
                 rng, progress_meter, args...)
        else
            func(atom, 0, chunk_size + left_over, n_patterns, n_train,
                 rng, progress_meter, args...)
        end
    end
end
@static if VERSION >= v"1.3.0-DEV.573"
    function _fit(res::CPUThreads, func, verbosity, stuff)
        atom, n, n_patterns, n_train, rng, progress_meter, args = stuff
        if verbosity > 0
            println("Ensemble-building in parallel on $(Threads.nthreads()) threads.")
        end
        nthreads = Threads.nthreads()
        chunk_size = div(n, nthreads)
        left_over = mod(n, nthreads)
        resvec = Vector(undef, nthreads) # FIXME: Make this type-stable?
        Threads.@threads for i = 1:nthreads
            resvec[i] = if i != nworkers()
                func(atom, 0, chunk_size, n_patterns, n_train,
                             rng, progress_meter, args...)
            else
                func(atom, 0, chunk_size + left_over, n_patterns, n_train,
                             rng, progress_meter, args...)
            end
        end
        return reduce(_reducer, resvec)
    end
end

function fit(model::EitherEnsembleModel{Atom},
             verbosity::Int, args...) where Atom<:Supervised

    X = args[1]
    y = args[2]
    if length(args) == 3
        w = args[3]
    else
        w = nothing
    end

    acceleration = model.acceleration
    if acceleration isa CPUProcesses && nworkers() == 1
        acceleration = default_resource()
    end

    if model.out_of_bag_measure isa Vector
        out_of_bag_measure = model.out_of_bag_measure
    else
        out_of_bag_measure = [model.out_of_bag_measure,]
    end

    if model.rng isa Integer
        rng = MersenneTwister(model.rng)
    else
        rng = model.rng
    end

    atom = model.atom
    n = model.n
    n_patterns = nrows(y)
    n_train = round(Int, floor(model.bagging_fraction*n_patterns))

    progress_meter = Progress(n, dt=0.5, desc="Training ensemble: ",
               barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    if !isempty(out_of_bag_measure)

        stuff = atom, n, n_patterns, n_train, rng, progress_meter, args
        ensemble, ensemble_indices =
            _fit(acceleration, get_ensemble_and_indices, verbosity, stuff)

    else

        stuff = atom, n, n_patterns, n_train, rng, progress_meter, args
        ensemble = _fit(acceleration, get_ensemble, verbosity, stuff)

    end

    fitresult = WrappedEnsemble(model.atom, ensemble)

    if !isempty(out_of_bag_measure)

        metrics=zeros(length(ensemble),length(out_of_bag_measure))
        for i= 1:length(ensemble)
            #oob indices
            ooB_indices=  setdiff(1:n_patterns, ensemble_indices[i])
            if isempty(ooB_indices)
                error("Empty out-of-bag sample. "*
                      "Data size too small or "*
                      "bagging_fraction too close to 1.0. ")
            end
            yhat = predict(atom, ensemble[i],
                           selectrows(X, ooB_indices))
            Xtest = selectrows(X, ooB_indices)
            ytest = selectrows(y, ooB_indices)
            if w === nothing
                wtest = nothing
            else
                wtest = selectrows(w, ooB_indices)
            end
            for k in eachindex(out_of_bag_measure)
                m = out_of_bag_measure[k]
                if reports_each_observation(m)
                    s =  aggregate(value(m, yhat, Xtest, ytest, wtest), m)
                else
                    s = value(m, yhat, Xtest, ytest, wtest)
                end
                metrics[i,k] = s
            end
        end

        # aggregate metrics across the ensembles:
        aggregated_metrics = map(eachindex(out_of_bag_measure)) do k
            aggregate(metrics[:,k], out_of_bag_measure[k])
        end

        names = Symbol.(string.(out_of_bag_measure))

    else
        aggregated_metrics = missing
    end

    report=(measures=out_of_bag_measure, oob_measurements=aggregated_metrics,)
    cache = deepcopy(model)

    return fitresult, cache, report

end

# if n is only parameter that changes, we just append to the existing
# ensemble, or truncate it:
function update(model::EitherEnsembleModel,
                verbosity::Int, fitresult, old_model, args...)

    n = model.n

    if MLJBase.is_same_except(model.atom, old_model.atom,
                              :n, :atomic_weights, :acceleration)
        if n > old_model.n
            verbosity < 1 ||
                @info "Building on existing ensemble of length $(old_model.n)"
            model.n = n - old_model.n # temporarily mutate the model
            wens, model_copy, report = fit(model, verbosity, args...)
            append!(fitresult.ensemble, wens.ensemble)
            model.n = n         # restore model
            model_copy.n = n    # new copy of the model
        else
            verbosity < 1 || @info "Truncating existing ensemble."
            fitresult.ensemble = fitresult.ensemble[1:n]
            model_copy = deepcopy(model)
        end
        cache, report = model_copy, NamedTuple()
        return fitresult, cache, report
    else
        return fit(model, verbosity, args...)
    end

end

function predict(model::EitherEnsembleModel, fitresult, Xnew)

    n = model.n
    if isempty(model.atomic_weights)
        atomic_weights = fill(1/n, n)
    else
        length(model.atomic_weights) == n ||
            error("Ensemble size and number of atomic_weights not the same.")
        atomic_weights = model.atomic_weights
    end
    predict(fitresult, atomic_weights, Xnew)
end

## METADATA

# Note: input and target traits are inherited from atom

MLJBase.supports_weights(::Type{<:EitherEnsembleModel{Atom}}) where Atom =
    MLJBase.supports_weights(Atom)

MLJBase.load_path(::Type{<:DeterministicEnsembleModel}) =
    "MLJ.DeterministicEnsembleModel"
MLJBase.package_name(::Type{<:DeterministicEnsembleModel}) = "MLJ"
MLJBase.package_uuid(::Type{<:DeterministicEnsembleModel}) = ""
MLJBase.package_url(::Type{<:DeterministicEnsembleModel}) =
    "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:DeterministicEnsembleModel{Atom}}) where Atom =
    MLJBase.is_pure_julia(Atom)
MLJBase.input_scitype(::Type{<:DeterministicEnsembleModel{Atom}}) where Atom =
    MLJBase.input_scitype(Atom)
MLJBase.target_scitype(::Type{<:DeterministicEnsembleModel{Atom}}) where Atom =
    MLJBase.target_scitype(Atom)

MLJBase.load_path(::Type{<:ProbabilisticEnsembleModel}) =
    "MLJ.ProbabilisticEnsembleModel"
MLJBase.package_name(::Type{<:ProbabilisticEnsembleModel}) = "MLJ"
MLJBase.package_uuid(::Type{<:ProbabilisticEnsembleModel}) = ""
MLJBase.package_url(::Type{<:ProbabilisticEnsembleModel}) =
    "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:ProbabilisticEnsembleModel{Atom}}) where Atom =
    MLJBase.is_pure_julia(Atom)
MLJBase.input_scitype(::Type{<:ProbabilisticEnsembleModel{Atom}}) where Atom =
    MLJBase.input_scitype(Atom)
MLJBase.target_scitype(::Type{<:ProbabilisticEnsembleModel{Atom}}) where Atom =
    MLJBase.target_scitype(Atom)
