const SupervisedNetwork = Union{DeterministicNetwork,ProbabilisticNetwork}

# to suppress inclusion in models():
MLJBase.is_wrapper(::Type{DeterministicNetwork}) = true
MLJBase.is_wrapper(::Type{ProbabilisticNetwork}) = true


## FALL-BACKS FOR LEARNING NETWORKS EXPORTED AS MODELS

function MLJBase.update(model::Union{SupervisedNetwork,UnsupervisedNetwork},
                        verbosity, yhat, cache, args...)

    # If any `model` field has been replaced (and not just mutated)
    # then we actually need to fit rather than update (which will
    # force build of a new learning network). If `model` has been
    # created using a learning network export macro, the test used
    # below is perfect. In any other case it is at least conservative:
    network_model_ids = objectid.(models(yhat))
    fields = [getproperty(model, name) for
        name in fieldnames(typeof(model))]
    submodels = filter(f->f isa Model, fields)
    submodel_ids = objectid.(submodels)
    if !issubset(submodel_ids, network_model_ids)
        return fit(model, verbosity, args...)
    end

    is_anonymised = cache isa NamedTuple{(:sources, :data)}

    if is_anonymised
        sources, data = cache.sources, cache.data
        for k in eachindex(sources)
            rebind!(sources[k], data[k])
        end
    end

    fit!(yhat; verbosity=verbosity)
    if is_anonymised
        for s in sources
            rebind!(s, nothing)
        end
    end

    return yhat, cache, nothing
end

MLJBase.predict(composite::SupervisedNetwork, fitresult, Xnew) =
    fitresult(Xnew)

MLJBase.transform(composite::UnsupervisedNetwork, fitresult, Xnew) =
    fitresult(Xnew)

function fitted_params(yhat::Node)
    machs = machines(yhat)
    fitted = [fitted_params(m) for m in machs]
    return (machines=machs, fitted_params=fitted)
end

fitted_params(composite::Union{SupervisedNetwork,UnsupervisedNetwork}, yhat) =
    fitted_params(yhat)


## FOR EXPORTING LEARNING NETWORKS BY HAND

"""
    anonymize!(sources...)

Returns a named tuple `(sources=..., data=....)` whose values are the
provided source nodes and their contents respectively, and clears the
contents of those source nodes.

"""
function anonymize!(sources...)
    data = Tuple(s.data for s in sources)
    [MLJ.rebind!(s, nothing) for s in sources]
    return (sources=sources, data=data)
end

function report(yhat::Node)
    machs = machines(yhat)
    reports = [report(m) for m in machs]
    return (machines=machs, reports=reports)
end

# what should be returned by a fit method for an exported learning
# network:
function fitresults(yhat)
    inputs = sources(yhat, kind=:input)
    targets = sources(yhat, kind=:target)
    length(inputs) == 1 ||
        error("Improperly exported supervised network does "*
              "not have a unique input source. ")
    if length(targets) == 1
        cache = anonymize!(inputs[1], targets[1])
    elseif length(targets) == 0
        cache = anonymize!(inputs[1])
    else
        error("Improperly exported network has multiple target sources. ")
    end
    r = report(yhat)
    return yhat, cache, r
end


## EXPORTING LEARNING NETWORKS AS MODELS WITH @from_network

"""
    replace(W::MLJ.Node, a1=>b1, a2=>b2, ...)

Create a deep copy of a node `W`, and thereby replicate the learning
network terminating at `W`, but replacing any specified sources and
models `a1, a2, ...` of the original network with `b1, b2, ...`.
"""
function Base.replace(W::Node, pairs::Pair...)

    # Note: We construct nodes of the new network as values of a
    # dictionary keyed on the nodes of the old network. Additionally,
    # there are dictionaries of models keyed on old models and
    # machines keyed on old machines. The node and machine
    # dictionaries must be built simultaneously.

    # build model dict:
    model_pairs = filter(collect(pairs)) do pair
        first(pair) isa Model
    end
    models_ = models(W)
    models_to_copy = setdiff(models_, first.(model_pairs))
    model_copy_pairs = [model=>deepcopy(model) for model in models_to_copy]
    newmodel_given_old = IdDict(vcat(model_pairs, model_copy_pairs))

    # build complete source replacement pairs:
    source_pairs = filter(collect(pairs)) do pair
        first(pair) isa Source
    end
    sources_ = sources(W)
    sources_to_copy = setdiff(sources_, first.(source_pairs))
    isempty(sources_to_copy) ||
        @warn "No replacement specified for one or more source nodes. "*
    "Data there will be duplicated. "
    source_copy_pairs = [source=>deepcopy(source) for source in sources_to_copy]
    all_source_pairs = vcat(source_pairs, source_copy_pairs)

    # drop source nodes from all nodes of network terminating at W:
    nodes_ = filter(nodes(W)) do N
        !(N isa Source)
    end
    isempty(nodes_) && error("All nodes in network are source nodes. ")
    # instantiate node and machine dictionaries:
    newnode_given_old =
        IdDict{AbstractNode,AbstractNode}(all_source_pairs)
    newmach_given_old = IdDict{NodalMachine,NodalMachine}()

    # build the new network:
    for N in nodes_
       args = [newnode_given_old[arg] for arg in N.args]
         if N.machine === nothing
             newnode_given_old[N] = node(N.operation, args...)
         else
             if N.machine in keys(newmach_given_old)
                 mach = newmach_given_old[N.machine]
             else
                 train_args = [newnode_given_old[arg] for arg in N.machine.args]
                 mach = machine(newmodel_given_old[N.machine.model], train_args...)
                 newmach_given_old[N.machine] = mach
             end
             newnode_given_old[N] = N.operation(mach, args...)
        end
    end

    return newnode_given_old[nodes_[end]]

 end

# closure for later:
function fit_method(network, models...)

    network_Xs = sources(network, kind=:input)[1]

    function fit(model::M, verbosity, X, y) where M <: Supervised
        replacement_models = [getproperty(model, fld)
                              for fld in fieldnames(M)]
        model_replacements = [models[j] => replacement_models[j]
                          for j in eachindex(models)]
        network_ys = sources(network, kind=:target)[1]
        Xs = source(X)
        ys = source(y, kind=:target)
        source_replacements = [network_Xs => Xs, network_ys => ys]
        replacements = vcat(model_replacements, source_replacements)
        yhat = replace(network, replacements...)

        Set([Xs, ys]) == Set(sources(yhat)) ||
            error("Failed to replace sources in network blueprint. ")

        fit!(yhat, verbosity=verbosity)

        return fitresults(yhat)
    end

    function fit(model::M, verbosity, X) where M <:Unsupervised
        replacement_models = [getproperty(model, fld)
                              for fld in fieldnames(M)]
        model_replacements = [models[j] => replacement_models[j]
                          for j in eachindex(models)]
        Xs = source(X)
        source_replacements = [network_Xs => Xs,]
        replacements = vcat(model_replacements, source_replacements)
        Xout = replace(network, replacements...)
        Set([Xs]) == Set(sources(Xout)) ||
            error("Failed to replace sources in network blueprint. ")

        fit!(Xout, verbosity=verbosity)

        return fitresults(Xout)
    end

    return fit

end

net_alert(message) = throw(ArgumentError("Learning network export error.\n"*
                                     string(message)))
net_alert(k::Int) = throw(ArgumentError("Learning network export error $k. "))

# returns Model supertype - or `missing` if arguments are incompatible
function kind_(is_supervised, is_probabilistic)
    if is_supervised
        if ismissing(is_probabilistic) || !is_probabilistic
            return :DeterministicNetwork
        else
            return :ProbabilisticNetwork
        end
    else
        if ismissing(is_probabilistic) || !is_probabilistic
            return :UnsupervisedNetwork
        else
            return missing
        end
    end
end

function from_network_preprocess(modl, ex,
                                 is_probabilistic::Union{Missing,Bool})

    ex isa Expr || net_alert(1)
    ex.head == :call || net_alert(2)
    ex.args[1] == :(<=) || net_alert(3)
    ex.args[2] isa Expr || net_alert(4)
    ex.args[2].head == :call || net_alert(5)
    modeltype_ex = ex.args[2].args[1]
    modeltype_ex isa Symbol || net_alert(6)
    if length(ex.args[2].args) == 1
        kw_exs = []
    else
        kw_exs = ex.args[2].args[2:end]
    end
    fieldname_exs = []
    model_exs = []
    for ex in kw_exs
        ex isa Expr || net_alert(7)
        ex.head == :kw || net_alert(8)
        variable_ex = ex.args[1]
        value_ex = ex.args[2]
        variable_ex isa Symbol || net_alert(9)
        push!(fieldname_exs, variable_ex)
        value = modl.eval(value_ex)
        value isa Model ||
            net_alert("Got $value but expected something of type `Model`.")
        push!(model_exs, value_ex)
    end
    N_ex = ex.args[3]
    N = modl.eval(N_ex)
    N isa AbstractNode ||
        net_alert("Got $N but expected something of type `AbstractNode`. ")

    inputs = sources(N, kind=:input)
    targets = sources(N, kind=:target)

    length(inputs) == 0 &&
        net_alert("Network has no source with `kind=:input`.")
    length(inputs) > 1  &&
        net_alert("Network has multiple sources with `kind=:input`.")
    length(targets) > 1 &&
        net_alert("Network has multiple sources with `kind=:target`.")

    is_supervised = length(targets) == 1

    kind = kind_(is_supervised, is_probabilistic)
    ismissing(kind) &&
        net_alert("Network appears unsupervised (has no source with "*
                  "`kind=:target`) and so `is_probabilistic=true` "*
                  "declaration is not allowed. ")

    models_ = [modl.eval(e) for e in model_exs]
    issubset(models_, models(N)) ||
        net_alert("One or more specified models are not in the learning network "*
              "terminating at $N_ex.\n Use models($N_ex) to inspect models. ")

    nodes_  = nodes(N)

    return modeltype_ex, fieldname_exs, model_exs, N_ex, kind

end

from_network_preprocess(modl, ex) = from_network_preprocess(modl, ex, missing)

function from_network_preprocess(modl, ex, kw_ex)
    kw_ex isa Expr || net_alert(10)
    kw_ex.head == :(=) || net_alert(11)
    kw_ex.args[1] == :is_probabilistic ||
        net_alert("Unrecognized keywork `$(kw_ex.args[1])`.")
    value = kw_ex.args[2]
    if value isa Bool
        return from_network_preprocess(modl, ex, value)
    else
        net_alert("`is_probabilistic` can only be `true` or `false`.")
    end
end

function from_network_(modl, modeltype_ex, fieldname_exs, model_exs,
                          N_ex, kind)

    args = gensym(:args)

    # code defining the composite model struct and fit method:
    program1 = quote

        mutable struct $modeltype_ex <: MLJ.$kind
            $(fieldname_exs...)
        end

        MLJ.fit(model::$modeltype_ex, verbosity::Integer, $args...) =
            MLJ.fit_method(
                $N_ex, $(model_exs...))(model, verbosity, $args...)

    end

    program2 = quote
        defaults =
            MLJ.@set_defaults $modeltype_ex deepcopy.([$(model_exs...)])

    end

    modl.eval(program1)
    modl.eval(program2)

end

"""
    @from_network(NewCompositeModel(fld1=model1, fld2=model2, ...) <= N
    @from_network(NewCompositeModel(fld1=model1, fld2=model2, ...) <= N is_probabilistic=false
    
Create a new stand-alone model type called `NewCompositeModel`, using
a learning network as a blueprint. Here `N` refers to the terminal
node of the learning network (from which final predictions or
transformations are fetched). 

*Important.* If the learning network is supervised (has a source with
`kind=:target`) and makes probabilistic predictions, then one must
declare `is_probabilistic=true`. In the deterministic case the keyword
argument can be omitted.

The model type `NewCompositeModel` is equipped with fields named
`:fld1`, `:fld2`, ..., which correspond to component models `model1`,
`model2`, ...,  appearing in the network (which must therefore be elements of
`models(N)`).  Deep copies of the specified component models are used
as default values in an automatically generated keyword constructor
for `NewCompositeModel`.

### Return value

 A new `NewCompositeModel` instance, with default field values.

For details and examples refer to the "Learning Networks" section of
the documentation.

"""
macro from_network(exs...)

    args = from_network_preprocess(__module__, exs...)
    modeltype_ex = args[1]

    from_network_(__module__, args...)

    esc(quote
        $modeltype_ex()
        end)

end


## A COMPOSITE FOR TESTING PURPOSES

"""
    SimpleDeterministicCompositeModel(;regressor=ConstantRegressor(),
                              transformer=FeatureSelector())

Construct a composite model consisting of a transformer
(`Unsupervised` model) followed by a `Deterministic` model. Mainly
intended for internal testing .

"""
mutable struct SimpleDeterministicCompositeModel{L<:Deterministic,
                             T<:Unsupervised} <: DeterministicNetwork
    model::L
    transformer::T

end

function SimpleDeterministicCompositeModel(; model=DeterministicConstantRegressor(),
                          transformer=FeatureSelector())

    composite =  SimpleDeterministicCompositeModel(model, transformer)

    message = MLJ.clean!(composite)
    isempty(message) || @warn message

    return composite

end

MLJBase.is_wrapper(::Type{<:SimpleDeterministicCompositeModel}) = true

function MLJBase.fit(composite::SimpleDeterministicCompositeModel,
                     verbosity::Integer, Xtrain, ytrain)
    X = source(Xtrain) # instantiates a source node
    y = source(ytrain, kind=:target)

    t = machine(composite.transformer, X)
    Xt = transform(t, X)

    l = machine(composite.model, Xt, y)
    yhat = predict(l, Xt)

    fit!(yhat, verbosity=verbosity)

    return fitresults(yhat)
end

# MLJBase.predict(composite::SimpleDeterministicCompositeModel, fitresult, Xnew) = fitresult(Xnew)

MLJBase.load_path(::Type{<:SimpleDeterministicCompositeModel}) = "MLJ.SimpleDeterministicCompositeModel"
MLJBase.package_name(::Type{<:SimpleDeterministicCompositeModel}) = "MLJ"
MLJBase.package_uuid(::Type{<:SimpleDeterministicCompositeModel}) = ""
MLJBase.package_url(::Type{<:SimpleDeterministicCompositeModel}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:SimpleDeterministicCompositeModel}) = true
MLJBase.input_scitype(::Type{<:SimpleDeterministicCompositeModel{L,T}}) where {L,T} =
    MLJBase.input_scitype(T)
MLJBase.target_scitype(::Type{<:SimpleDeterministicCompositeModel{L,T}}) where {L,T} =
    MLJBase.target_scitype(L)
