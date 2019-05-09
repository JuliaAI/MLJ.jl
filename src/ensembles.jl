# shorter display of MersenneTwister:
Base.show(stream::IO, t::Random.MersenneTwister) =
    print(stream, "MersenneTwister($(t.seed))")


## WEIGHTED ENSEMBLES OF FITRESULTS

# Atom is atomic model type, eg, DecisionTree
# R will be the tightest type of the atom fit-results.
using StatsBase
mutable struct WrappedEnsemble{R,Atom <: Supervised} <: MLJType
    atom::Atom
    ensemble::Vector{R}
end

# to enable trait-based dispatch of predict:
predict(wens::WrappedEnsemble{R,Atom}, weights, Xnew) where {R,Atom<:Deterministic} =
    predict(wens, weights, Xnew, Deterministic, target_scitype_union(Atom))
predict(wens::WrappedEnsemble{R,Atom}, weights, Xnew) where {R,Atom<:Probabilistic} =
    predict(wens, weights, Xnew, Probabilistic, target_scitype_union(Atom))

function predict(wens::WrappedEnsemble,
                 weights,
                 Xnew,
                 ::Type{Deterministic}, ::Type{<:Finite})

    # weights ignored in this case

    ensemble = wens.ensemble
    atom = wens.atom

    n_atoms = length(ensemble)

    n_atoms > 0  || @error "Empty ensemble cannot make predictions."

    # TODO: make this more memory efficient but note that the type of
    # Xnew is unknown (ie, model dependent)
    predictions =
        reduce(hcat, [predict(atom, fitresult, Xnew) for fitresult in ensemble])
    classes = levels(predictions)
    n = size(predictions, 1)
    prediction =
        categorical(vcat([mode(predictions[i,:]) for i in 1:n], classes))[1:n]
    return prediction
end

function predict(wens::WrappedEnsemble, weights, Xnew, ::Type{Deterministic}, ::Type{<:Continuous})
    ensemble = wens.ensemble

    atom = wens.atom

    n_atoms = length(ensemble)

    n_atoms > 0  || @error "Empty ensemble cannot make predictions."

    # TODO: make more memory efficient:
    predictions = reduce(hcat, [weights[k]*predict(atom, ensemble[k], Xnew) for k in 1:n_atoms])
    prediction =  [sum(predictions[i,:]) for i in 1:size(predictions, 1)]

    return prediction
end

function predict(wens::WrappedEnsemble, weights, Xnew, ::Type{Probabilistic}, ::Type{<:Finite})

    ensemble = wens.ensemble

    atom = wens.atom

    n_atoms = length(ensemble)

    n_atoms > 0  || @error "Empty ensemble cannot make predictions."

    # TODO: make this more memory efficient but note that the type of
    # Xnew is unknown (ie, model dependent):

    # a matrix of probability distributions:
    predictions = reduce(hcat, [predict(atom, fitresult, Xnew) for fitresult in ensemble])
    n_rows = size(predictions, 1)

    # the weighted averages over the ensemble of the discrete pdf's:
    predictions  = [MLJBase.average([predictions[i,k] for k in 1:n_atoms], weights=weights) for i in 1:n_rows]

    return predictions
end

function predict(wens::WrappedEnsemble, weights, Xnew, ::Type{Probabilistic}, ::Type{<:Continuous})

    ensemble = wens.ensemble

    atom = wens.atom

    n_atoms = length(ensemble)

    n_atoms > 0  || @error "Empty ensemble cannot make predictions."

    # TODO: make this more memory efficient but note that the type of
    # Xnew is unknown (ie, model dependent):

    # a matrix of probability distributions:
    predictions = reduce(hcat, [predict(atom, fitresult, Xnew) for fitresult in ensemble])

    # n_rows = size(predictions, 1)
    # # the weighted average over the ensemble of the pdf means and pdf variances:
    # μs  = [sum([weights[k]*mean(predictions[i,k]) for k in 1:n_atoms]) for i in 1:n_rows]
    # σ2s = [sum([weights[k]*var(predictions[i,k]) for k in 1:n_atoms]) for i in 1:n_rows]

    # # a vector of normal probability distributions:
    # prediction = [Distributions.Normal(μs[i], sqrt(σ2s[i])) for i in 1:n_rows]

    prediction = [Distributions.MixtureModel(predictions[i,:], weights) for i in 1:size(predictions, 1)]

    return prediction

end


## CORE ENSEMBLE-BUILDING FUNCTIONS

# for when out-of-bag performance estimates are requested:
function get_ensemble_and_indices(atom::Supervised, verbosity, X, y, n, n_patterns,
                      n_train, rng, progress_meter)
    
    ensemble_indices = [StatsBase.sample(rng, 1:n_patterns, n_train, replace=false)
                        for i in 1:n]
    ensemble = map(ensemble_indices) do train_rows
        verbosity < 1 || next!(progress_meter)
        atom_fitresult, atom_cache, atom_report =
            fit(atom, verbosity - 1, selectrows(X, train_rows), selectrows(y, train_rows))
        atom_fitresult
    end
    verbosity < 1 || println()

    return (ensemble, ensemble_indices)

end

# for when out-of-bag performance estimates are not requested:
function get_ensemble(atom::Supervised, verbosity, X, y, n, n_patterns,
                      n_train, rng, progress_meter)

    # define generator of training rows:
    ensemble_indices = (StatsBase.sample(rng, 1:n_patterns, n_train, replace=false)
                        for i in 1:n)
    ensemble = map(ensemble_indices) do train_rows
        verbosity < 1 || next!(progress_meter)
        atom_fitresult, atom_cache, atom_report =
            fit(atom, verbosity - 1, selectrows(X, train_rows), selectrows(y, train_rows))
        atom_fitresult
    end
    verbosity < 1 || println()

    return ensemble

end

pair_vcat(p, q) = (vcat(p[1], q[1]), vcat(p[2], q[2]))
    

## ENSEMBLE MODEL FOR DETERMINISTIC MODELS

mutable struct DeterministicEnsembleModel{Atom<:Deterministic} <: Deterministic
    atom::Atom
    weights::Vector{Float64}
    bagging_fraction::Float64
    rng::Union{Int,AbstractRNG}
    n::Int
    parallel::Bool
    out_of_bag_measure # TODO: type this
end

function clean!(model::DeterministicEnsembleModel) 

    message = ""

    if model.bagging_fraction > 1 || model.bagging_fraction <= 0
        message = message*"`bagging_fraction` should be "*
        "in the range (0,1]. Reset to 1. "
        model.bagging_fraction = 1.0
    end
    if target_scitype_union(model.atom)<:Finite && !isempty(model.weights)
        message = message*"weights will be ignored to form predictions. "
    elseif !isempty(model.weights)
        total = sum(model.weights)
        if !(total ≈ 1.0)
            message = message*"weights should sum to one and are being automatically normalized. "
            model.weights = model.weights/total
        end
    end

    return message

end

# constructor to infer type automatically:
DeterministicEnsembleModel(atom::Atom, weights,
                           bagging_fraction, rng, n, parallel, out_of_bag_measure) where Atom<:Deterministic =
                               DeterministicEnsembleModel{Atom}(atom, weights,
                                                                   bagging_fraction, rng, n, parallel, out_of_bag_measure)

# lazy keyword constructors:
function DeterministicEnsembleModel(;atom=DeterministicConstantClassifier(),
                                    weights=Float64[],
                                    bagging_fraction=0.8,
                                    rng=Random.GLOBAL_RNG,
                                    n::Int=100,
                                    parallel=true,
                                    out_of_bag_measure=[])

    model = DeterministicEnsembleModel(atom, weights, bagging_fraction, rng,
                                       n, parallel, out_of_bag_measure)

    message = clean!(model)
    isempty(message) || @warn message

    return model
end


## ENSEMBLE MODEL FOR PROBABILISTIC MODELS

mutable struct ProbabilisticEnsembleModel{Atom<:Probabilistic} <: Probabilistic
    atom::Atom
    weights::Vector{Float64}
    bagging_fraction::Float64
    rng::Union{Int, AbstractRNG}
    n::Int
    parallel::Bool
    out_of_bag_measure
end

function clean!(model::ProbabilisticEnsembleModel) 

    message = ""

    if model.bagging_fraction > 1 || model.bagging_fraction <= 0
        message = message*"`bagging_fraction` should be "*
        "in the range (0,1]. Reset to 1. "
        model.bagging_fraction = 1.0
    end

    if !isempty(model.weights)
        total = sum(model.weights)
        if !(total ≈ 1.0)
            message = message*"Weights should sum to one and are being automatically normalized. "
            model.weights = model.weights/total
        end
    end

    return message

end

# constructor to infer type automatically:
ProbabilisticEnsembleModel(atom::Atom, weights, bagging_fraction, rng, n, parallel, out_of_bag_measure) where Atom<:Probabilistic =
                               ProbabilisticEnsembleModel{Atom}(atom, weights, bagging_fraction, rng, n, parallel, out_of_bag_measure)

# lazy keyword constructor:
function ProbabilisticEnsembleModel(;atom=ConstantProbabilisticClassifier(),
                                    weights=Float64[],
                                    bagging_fraction=0.8,
                                    rng=Random.GLOBAL_RNG,
                                    n::Int=100,
                                    parallel=true,
                                    out_of_bag_measure=[])

    model = ProbabilisticEnsembleModel(atom, weights, bagging_fraction, rng, n, parallel,out_of_bag_measure)

    message = clean!(model)
    isempty(message) || @warn message

    return model
end


## COMMON CONSTRUCTOR

"""
    EnsembleModel(atom=nothing, 
                  weights=Float64[],
                  bagging_fraction=0.8,
                  rng=GLOBAL_RNG, n=100,
                  parallel=true,
                  out_of_bag_measure=[])

Create a model for training an ensemble of `n` learners, with optional
bagging, each with associated model `atom`. Ensembling is useful if
`fit!(machine(atom, data...))` does not create identical models on
repeated calls (ie, is a stochastic model, such as a decision tree
with randomized node selection criteria), or if `bagging_fraction` is
set to a value less than 1.0, or both. The constructor fails if no
`atom` is specified.

If `rng` is an integer, then `MersenneTwister(rng)` is the random
number generator used for bagging. Otherwise some `AbstractRNG` object
is expected.

Predictions are weighted according to the vector `weights` (to allow
for external optimization) except in the case that `atom` is a
`Deterministic` classifier. Uniform weights are used if `weight` has
zero length.

The ensemble model is `Deterministic` or `Probabilistic`, according to
the corresponding supertype of `atom`. In the case of deterministic classifiers
(`target_scitype_union(atom) <: Finite`), the
predictions are majority votes, and for regressors
(`target_scitype_union(atom)<: Continuous`) they are ordinary averages.
Probabilistic predictions are obtained by averaging the atomic
probability distribution/mass functions; in particular, for regressors, the
ensemble prediction on each input pattern has the type
`MixtureModel{VF,VS,D}` from the Distributions.jl package, where `D`
is the type of predicted distribution for `atom`.

If a single measure or non-empty vector of measusres is specified by
`out_of_bag_measure`, then out-of-bag estimates of performance are
reported.

"""
function EnsembleModel(; args...)
    d = Dict(args)
    :atom in keys(d) || error("No atomic model specified. Use EnsembleModel(atom=...)")
    if d[:atom] isa Deterministic
        return DeterministicEnsembleModel(; d...)
    elseif d[:atom] isa Probabilistic
        return ProbabilisticEnsembleModel(; d...)
    end
    error("$(d[:atom]) does not appear to be a Supervised model.")
end


## THE COMMON FIT AND PREDICT METHODS

const EitherEnsembleModel{Atom} = Union{DeterministicEnsembleModel{Atom}, ProbabilisticEnsembleModel{Atom}}

MLJBase.is_wrapper(::Type{<:EitherEnsembleModel}) = true

function fit(model::EitherEnsembleModel{Atom}, verbosity::Int, X, y) where Atom<:Supervised

    parallel = model.parallel

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
    n_patterns = nrows(X)
    n_train = round(Int, floor(model.bagging_fraction*n_patterns))

    progress_meter = Progress(n, dt=0.5, desc="Training ensemble: ",
                              barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)


    if !isempty(out_of_bag_measure)
        
        if !parallel || nworkers() == 1 # build in serial
            ensemble, ensemble_indices = get_ensemble_and_indices(atom, verbosity, X, y,
                                                   n, n_patterns, n_train, rng, progress_meter)
        else # build in parallel
            if verbosity > 0
                println("Ensemble-building in parallel on $(nworkers()) processors.")
            end
            chunk_size = div(n, nworkers())
            left_over = mod(n, nworkers())
            ensemble, ensemble_indices =  @distributed (pair_vcat) for i = 1:nworkers()
                if i != nworkers()
                    get_ensemble_and_indices(atom, 0, X, y, chunk_size, n_patterns, n_train,
                                 rng, progress_meter)
                else
                    get_ensemble_and_indices(atom, 0, X, y, chunk_size + left_over, n_patterns, n_train,
                                 rng, progress_meter)
                end
            end
        end

    else
        
        if !parallel || nworkers() == 1 # build in serial
            ensemble = get_ensemble(atom, verbosity, X, y,
                                    n, n_patterns, n_train, rng, progress_meter)
        else # build in parallel
            if verbosity > 0
                println("Ensemble-building in parallel on $(nworkers()) processors.")
            end
            chunk_size = div(n, nworkers())
            left_over = mod(n, nworkers())
            ensemble =  @distributed (vcat) for i = 1:nworkers()
                if i != nworkers()
                    get_ensemble(atom, 0, X, y, chunk_size, n_patterns, n_train,
                                 rng, progress_meter)
                else
                    get_ensemble(atom, 0, X, y, chunk_size + left_over, n_patterns, n_train,
                                 rng, progress_meter)
                end
            end
        end
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
            predictions = predict(atom, ensemble[i], selectrows(X,ooB_indices))

            for k in eachindex(out_of_bag_measure)
                metrics[i,k] = out_of_bag_measure[k](predictions,selectrows(y, ooB_indices))
            end

        end
        metrics=mean(metrics, dims=1)

        # Anthony thinks inclusion of this code is wrong:
        # else TODO
        #     @show hcat([metrics[k,:]*model.weights[k] for k=1:length(ensemble)]...)
        #     metrics=mean(hcat([metrics[k,:]*model.weights[k] for k=1:length(ensemble)]...),dims=1)
        # end

        names = Symbol.(string.(out_of_bag_measure))
        oob_estimates=NamedTuple{Tuple(names)}(Tuple(vec(metrics)))

    else
        oob_estimates=NamedTuple()
    end

    report=(oob_estimates=oob_estimates,)
    cache = deepcopy(model)

    return fitresult, cache, report

end

# if n is only parameter that changes, we just append to the existing
# ensemble, or truncate it:
function update(model::EitherEnsembleModel, verbosity::Int, fitresult, old_model, X, y)

    n = model.n

    if model.atom == old_model.atom &&
        model.bagging_fraction == old_model.bagging_fraction
        if n > old_model.n
            verbosity < 1 || @info "Building on existing ensemble of length $(old_model.n)"
            model.n = n - old_model.n # temporarily mutate the model
            wens, model_copy, report = fit(model, verbosity, X, y)
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
        return fit(model, verbosity, X, y)
    end

end

function predict(model::EitherEnsembleModel, fitresult, Xnew)

    # because weights could have changed since last fit:
    message = clean!(model)
    isempty(message) || @warn message

    n = model.n
    if isempty(model.weights)
        weights = fill(1/n, n)
    else
        length(model.weights) == n || error("Ensemble size and number of weights not the same.")
        weights = model.weights
    end
    predict(fitresult, weights, Xnew)
end

## METADATA

# Note: input and target traits are inherited from atom

MLJBase.load_path(::Type{<:DeterministicEnsembleModel}) = "MLJ.DeterministicEnsembleModel"
MLJBase.package_name(::Type{<:DeterministicEnsembleModel}) = "MLJ"
MLJBase.package_uuid(::Type{<:DeterministicEnsembleModel}) = ""
MLJBase.package_url(::Type{<:DeterministicEnsembleModel}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:DeterministicEnsembleModel{Atom}}) where Atom = MLJBase.is_pure_julia(Atom)
MLJBase.input_scitype_union(::Type{<:DeterministicEnsembleModel{Atom}}) where Atom = MLJBase.input_scitype_union(Atom)
MLJBase.target_scitype_union(::Type{<:DeterministicEnsembleModel{Atom}}) where Atom = MLJBase.target_scitype_union(Atom)
MLJBase.input_is_multivariate(::Type{<:DeterministicEnsembleModel{Atom}}) where Atom = MLJBase.input_is_multivariate(Atom)

MLJBase.load_path(::Type{<:ProbabilisticEnsembleModel}) = "MLJ.ProbabilisticEnsembleModel"
MLJBase.package_name(::Type{<:ProbabilisticEnsembleModel}) = "MLJ"
MLJBase.package_uuid(::Type{<:ProbabilisticEnsembleModel}) = ""
MLJBase.package_url(::Type{<:ProbabilisticEnsembleModel}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:ProbabilisticEnsembleModel{Atom}}) where Atom = MLJBase.is_pure_julia(Atom)
MLJBase.input_scitype_union(::Type{<:ProbabilisticEnsembleModel{Atom}}) where Atom = MLJBase.input_scitype_union(Atom)
MLJBase.target_scitype_union(::Type{<:ProbabilisticEnsembleModel{Atom}}) where Atom = MLJBase.target_scitype_union(Atom)
MLJBase.input_is_multivariate(::Type{<:ProbabilisticEnsembleModel{Atom}}) where Atom = MLJBase.input_is_multivariate(Atom)

### old KoalaEnsembles code for optimizing the weights in the deterministic regressor case:

#     # Optimize weights:

#     n = length(ensemble)

#     if model.weight_regularization == 1
#         weights = ones(n)/n
#         verbosity < 1 || @info "Weighting atoms uniformly."
#     else
#         verbosity < 1 || print("\nOptimizing weights...")
#         Y = Array{Float64}(undef, n, n_patterns)
#         for k in 1:n
#             Y[k,:] = predict(model.atom, ensemble[k], X, false, false)
#         end

#         # If I rescale all predictions by the same amount it makes no
#         # difference to the values of the optimal weights:
#         ybar = mean(abs.(y))
#         Y = Y/ybar

#         A = Y*Y'
#         b = Y*(y/ybar)

#         scale = abs(det(A))^(1/n)

#         if scale < eps(Float64)

#             verbosity < 0 || @warn "Weight optimization problem ill-conditioned. " *
#                  "Using uniform weights."
#             weights = ones(n)/n

#         else

#             # need regularization, `gamma`, between 0 and infinity:
#             if model.weight_regularization == 0
#                 gamma = 0
#             else
#                 gamma = exp(atanh(2*model.weight_regularization - 1))
#             end

#             # add regularization and augment linear system for constraint
#             # (weights sum to one)
#             AA = hcat(A + scale*gamma*Matrix(I, n, n), ones(n))
#             AA = vcat(AA, vcat(ones(n), [0.0])')
#             bb = b + scale*gamma*ones(n)/n
#             bb = vcat(bb, [1.0])

#             weights = (AA \ bb)[1:n] # drop Lagrange multiplier
#             verbosity < 1 || println("\r$n weights optimized.\n")

#         end

#     end

#     fitresult = WrappedEnsemble(model.atom, ensemble, weights)
#     report = Dict{Symbol, Any}()
#     report[falsermalized_weights] = weights*length(weights)

#     cache = (X, y, scheme_X, ensemble)

#     return fitresult, report, cache

# end

# predict(model::ProbabilisticEnsembleModel, fitresult, Xt, parallel, verbosity) =
#     predict(fitresult, Xt)

# function fit_weights!(mach::SupervisedMachine{WrappedEnsemble{R, Atom},
#                                               ProbabilisticEnsembleModel{R, Atom}};
#               verbosity=1, parallel=true) where {R, Atom <: Supervised{R}}

#     mach.n_iter != 0 || @error "Cannot fit weights to empty ensemble."

#     mach.fitresult, report, mach.cache =
#         fit(mach.model, mach.cache, false, parallel, verbosity;
#             optimize_weights_only=true)
#     merge!(mach.report, report)

#     return mach
# end

# function weight_regularization_curve(mach::SupervisedMachine{WrappedEnsemble{R, Atom},
#                                                            ProbabilisticEnsembleModel{R, Atom}},
#                                      test_rows;
#                                      verbosity=1, parallel=true,
#                                      range=range(0, stop=1, length=101),
#                                      raw=false) where {R, Atom <: Supervised{R}}

#     mach.n_iter > 0 || @error "No atoms in the ensemble. Run `fit!` first."
#     !raw || verbosity < 0 ||
#         @warn "Reporting errors for *transformed* target. Use `raw=false` "*
#              " to report true errors."

#     if parallel && nworkers() > 1
#         if verbosity >= 1
#             println("Optimizing weights in parallel on $(nworkers()) processors.")
#         end
#         errors = pmap(range) do w
#             verbosity < 2 || print("\rweight_regularization=$w       ")
#             mach.model.weight_regularization = w
#             fit_weights!(mach; parallel=false, verbosity=verbosity - 1)
#             err(mach, test_rows, raw=raw)
#         end
#     else
#         errors = Float64[]
#         for w in range
#             verbosity < 1 || print("\rweight_regularization=$w       ")
#             mach.model.weight_regularization = w
#             fit_weights!(mach; parallel= parallel, verbosity=verbosity - 1)
#             push!(errors, err(mach, test_rows, raw=raw))
#         end
#         verbosity < 1 || println()

#         mach.report[:weight_regularization_curve] = (range, errors)
#     end

#     return range, errors
# end
