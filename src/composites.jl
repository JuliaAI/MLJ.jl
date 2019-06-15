const SupervisedNetwork = Union{DeterministicNetwork,ProbabilisticNetwork}

# to suppress inclusion in models():
MLJBase.is_wrapper(::Type{DeterministicNetwork}) = true
MLJBase.is_wrapper(::Type{ProbabilisticNetwork}) = true

# fall-back for updating learning networks exported as models:
function MLJBase.update(model::Union{SupervisedNetwork,UnsupervisedNetwork},
                        verbosity, fitresult, cache, args...)
    fit!(fitresult; verbosity=verbosity)
    return fitresult, cache, nothing
end

# fall-back for predicting on learning networks exported as models
MLJBase.predict(composite::SupervisedNetwork, fitresult, Xnew) =
    fitresult(Xnew)

"""
    MLJ.tree(N::Node)

Return a description of the tree defined by the learning network
terminating at node `N`. 

"""
tree(s::MLJ.Source) = (source = s,)
function tree(W::MLJ.Node)
    mach = W.machine
    if mach == nothing
        value2 = nothing
        endkeys=[]
        endvalues=[]
    else
        value2 = mach.model        
        endkeys = [Symbol("train_arg", i) for i in eachindex(mach.args)]
        endvalues = [tree(arg) for arg in mach.args]
    end
    keys = tuple(:operation,  :model,
                 [Symbol("arg", i) for i in eachindex(W.args)]...,
                 endkeys...)
    values = tuple(W.operation, value2,
                   [tree(arg) for arg in W.args]...,
                   endvalues...)
    return NamedTuple{keys}(values)
end

# get the top level args of the tree of some node:
function args(tree) 
    keys_ = filter(keys(tree) |> collect) do key
        match(r"^arg[0-9]*", string(key)) != nothing
    end
    return [getproperty(tree, key) for key in keys_]
end
        
# get the top level train_args of the tree of some node:
function train_args(tree) 
    keys_ = filter(keys(tree) |> collect) do key
        match(r"^train_arg[0-9]*", string(key)) != nothing
    end
    return [getproperty(tree, key) for key in keys_]
end    
        
"""

    models(N::AbstractNode)

A vector of all models referenced by node `N`, each model
appearing exactly once.

"""
function models(W::MLJ.AbstractNode)
    models_ = filter(flat_values(tree(W)) |> collect) do model
        model isa MLJ.Model
    end
    return unique(models_)
end

"""
   sources(N::AbstractNode)

A vector of all sources referenced by calls `N()` and `fit!(N)`. These
are the sources of the directed acyclic graph associated with the
learning network terminating at `N`.

Not to be confused with `origins(N)` which refers to the same graph with edges corresponding to training arguments deleted.

See also: orgins, source

"""
function sources(W::MLJ.AbstractNode)
    sources_ = filter(MLJ.flat_values(tree(W)) |> collect) do model
        model isa MLJ.Source
    end
    return unique(sources_)
end

""" 
    machines(N)

List all machines in the learning network terminating at node `N`.

"""
machines(W::MLJ.Source) = Any[]
function machines(W::MLJ.Node)
    if W.machine == nothing
        return vcat([machines(arg) for arg in W.args]...) |> unique
    else
        return vcat(Any[W.machine, ],
                   [machines(arg) for arg in W.args]..., 
                   [machines(arg) for arg in W.machine.args]...) |> unique
    end
end

"""
    replace(W::MLJ.Node, a1=>b1, a2=>b2, ....)

Create a deep copy of a node `W`, and thereby replicate the learning
network terminating at `W`, but replacing any specified sources and
models `a1, a2, ...` of the original network with the specified targets
`b1, b2, ...`.

"""
function Base.replace(W::Node, pairs::Pair...) where N

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
    source_copy_pairs = [source=>deepcopy(source) for source in sources_to_copy]
    all_source_pairs = vcat(source_pairs, source_copy_pairs)

    # drop source nodes from all nodes of network terminating at W:
    nodes_ = filter(nodes(W)) do N
        !(N isa Source)
    end
    
    # instantiate node and machine dictionaries:
    newnode_given_old =
        IdDict{AbstractNode,AbstractNode}(all_source_pairs) 
    newmach_given_old = IdDict{NodalMachine,NodalMachine}()

    # build the new network:
    for N in nodes_
        args = [newnode_given_old[arg] for arg in N.args]
        if N.machine == nothing
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

"""
    reset!(N::Node)

Place the learning network terminating at node `N` into a state in
which `fit!(N)` will retrain from scratch all machines in its
dependency tape. Does not actually train any machine or alter
fit-results. (The method simply resets `m.state` to zero, for every
machine `m` in the network.)

"""
function reset!(W::Node)
    for mach in machines(W)
        mach.state = 0 # to do: replace with dagger object
    end
end

# closures for later:
function supervised_fit_method(network_Xs, network_ys, network_N,
                               network_models...)

    function fit(model::M, verbosity, X, y) where M <:Supervised
        Xs = source(X)
        ys = source(y)
        replacement_models = [getproperty(model, fld)
                              for fld in fieldnames(M)]
        model_replacements = [network_models[j] => replacement_models[j]
                              for j in eachindex(network_models)]
        source_replacements = [network_Xs => Xs, network_ys => ys]
        replacements = vcat(model_replacements, source_replacements)
        yhat = replace(network_N, replacements...)

        Set([Xs, ys]) == Set(sources(yhat)) ||
            error("Failed to replace sources in network blueprint. ")

        fit!(yhat, verbosity=verbosity)
        cache = nothing
        report = nothing
        return yhat, cache, report
    end

    return fit
end
function unsupervised_fit_method(network_Xs, network_N,
                               network_models...)

    function fit(model::M, verbosity, X) where M <:Unsupervised
        Xs = source(X)
        replacement_models = [getproperty(model, fld)
                              for fld in fieldnames(M)]
        model_replacements = [network_models[j] => replacement_models[j]
                              for j in eachindex(network_models)]
        source_replacements = [network_Xs => Xs,]
        replacements = vcat(model_replacements, source_replacements)
        Xout = replace(network_N, replacements...)
        Set([Xs]) == Set(sources(Xout)) ||
            error("Failed to replace sources in network blueprint. ")
        fit!(Xout, verbosity=verbosity)
        cache = nothing
        report = nothing
        return Xout, cache, report
    end

    return fit
end
        
"""
    
   @from_network NewCompositeModel(fld1=model1, fld2=model2, ...) <= (Xs, N)
   @from_network NewCompositeModel(fld1=model1, fld2=model2, ...) <= (Xs, ys, N)
   
Create, respectively, a new stand-alone unsupervised or superivsed
model type `NewCompositeModel` using a learning network as a
blueprint. Here `Xs`, `ys` and `N` refer to the input source, node,
target source node and terminating source node of the network. The
model type `NewCompositeModel` is equipped with fields named `:fld1`,
`:fld2`, ..., which correspond to component models `model1`, `model2`
appearing in the network (which must therefore be elements of
`models(N)`).  Deep copies of the specified component models are used
as default values in an automatically generated keyword constructor
for `NewCompositeModel`.

Return value: A new `NewCompositeModel` instance, with default
field values.

For details and examples refer to the "Learning Networks" section of
the documentation.

"""
macro from_network(ex)
    modeltype_ex = ex.args[2].args[1]
    kw_exs = ex.args[2].args[2:end]
    fieldname_exs = [k.args[1] for k in kw_exs]
    model_exs = [k.args[2] for k in kw_exs] 
    Xs_ex = ex.args[3].args[1]   # input node
    N_ex = ex.args[3].args[end]  # output node

    # TODO: add more type and syntax checks here:
    
    N = __module__.eval(N_ex)
    N isa Node ||
        error("$(typeof(N)) given where Node was expected. ")

    models_ = [__module__.eval(e) for e in model_exs]
    issubset(models_, models(N)) ||
        error("One or more specified models not in the learning network "*
              "terminating at $N_ex.\n Use models($N_ex) to inspect models. ")

    nodes_  = nodes(N)
    Xs = __module__.eval(Xs_ex)
    Xs in nodes_ ||
        error("Specified input source $Xs_ex is not a source of $N_ex.")
    
    if length(ex.args[3].args) == 3
        ys_ex = ex.args[3].args[2] # target node
        ys = __module__.eval(ys_ex) 
        ys in nodes_ ||
        error("Specified target source $ys_ex is not a source of $N_ex.")
        from_network_(__module__, modeltype_ex, fieldname_exs, model_exs,
                   Xs_ex, ys_ex, N_ex)
    else
        from_network_(__module__, modeltype_ex, fieldname_exs, model_exs,
                   Xs_ex, N_ex)
    end
    esc(quote
        $modeltype_ex()
        end)
end

# supervised case:
function from_network_(mod, modeltype_ex, fieldname_exs, model_exs,
                    Xs_ex, ys_ex, N_ex)

    N = mod.eval(N_ex)
    if MLJBase.is_probabilistic(typeof(models(N)[1]))
        subtype_ex = :ProbabilisticNetwork
    else
        subtype_ex = :DeterministicNetwork
    end

    XX = gensym(:X)
    yy = gensym(:y)
    
    # code defining the composite model struct and fit method:
    program1 = quote
        
        mutable struct $modeltype_ex <: MLJ.$subtype_ex
            $(fieldname_exs...)
        end
        
        MLJ.fit(model::$modeltype_ex, verbosity::Integer, $XX, $yy) =
            MLJ.supervised_fit_method($Xs_ex, $ys_ex, $N_ex,
                            $(model_exs...))(model, verbosity, $XX, $yy)
    end
    
    program2 = quote
        defaults = 
            MLJ.@set_defaults $modeltype_ex deepcopy.([$(model_exs...)])

    end

    mod.eval(program1)   
    mod.eval(program2)

end

# unsupervised case:
function from_network_(mod, modeltype_ex, fieldname_exs, model_exs,
                    Xs_ex, N_ex)

    subtype_ex = :UnsupervisedNetwork

    XX = gensym(:X)
    
    # code defining the composite model struct and fit method:
    program1 = quote
        
        mutable struct $modeltype_ex <: MLJ.$subtype_ex
            $(fieldname_exs...)
        end
        
        MLJ.fit(model::$modeltype_ex, verbosity::Integer, $XX) =
            MLJ.unsupervised_fit_method($Xs_ex, $N_ex,
                                      $(model_exs...))(model, verbosity, $XX)
    end
    
    program2 = quote
        defaults = 
        MLJ.@set_defaults $modeltype_ex deepcopy.([$(model_exs...)])
    end

    mod.eval(program1)   
    mod.eval(program2)

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

function MLJBase.fit(composite::SimpleDeterministicCompositeModel, verbosity::Int, Xtrain, ytrain)
    X = source(Xtrain) # instantiates a source node
    y = source(ytrain)

    t = machine(composite.transformer, X)
    Xt = transform(t, X)

    l = machine(composite.model, Xt, y)
    yhat = predict(l, Xt)

    fit!(yhat, verbosity=verbosity)
    fitresult = yhat
    report = l.report
    cache = l
    return fitresult, cache, report
end

# MLJBase.predict(composite::SimpleDeterministicCompositeModel, fitresult, Xnew) = fitresult(Xnew)

MLJBase.load_path(::Type{<:SimpleDeterministicCompositeModel}) = "MLJ.SimpleDeterministicCompositeModel"
MLJBase.package_name(::Type{<:SimpleDeterministicCompositeModel}) = "MLJ"
MLJBase.package_uuid(::Type{<:SimpleDeterministicCompositeModel}) = ""
MLJBase.package_url(::Type{<:SimpleDeterministicCompositeModel}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:SimpleDeterministicCompositeModel}) = true
# MLJBase.input_scitype_union(::Type{<:SimpleDeterministicCompositeModel}) = 
# MLJBase.target_scitype_union(::Type{<:SimpleDeterministicCompositeModel}) = 

