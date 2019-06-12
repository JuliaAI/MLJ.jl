const SupervisedNetwork = Union{DeterministicNetwork,ProbabilisticNetwork}

# to suppress inclusion in models():
MLJBase.is_wrapper(::Type{DeterministicNetwork}) = true
MLJBase.is_wrapper(::Type{ProbabilisticNetwork}) = true

# fall-back for updating learning networks exported as models:
function MLJBase.update(model::SupervisedNetwork, verbosity, fitresult, cache, args...)
    fit!(fitresult; verbosity=verbosity)
    return fitresult, cache, nothing
end

# fall-back for predicting on learning networks exported as models
MLJBase.predict(composite::SupervisedNetwork, fitresult, Xnew) =
    fitresult(Xnew)

"""
    MLJ.tree(N::Node)

Return a tree-like summary of the learning network terminating at node
`N`.

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

# similar to tree but returns arguments as vectors, rather than individually
tree2(s::MLJ.Source) = (source = s,)
function tree2(W::MLJ.Node)
    mach = W.machine
    if mach == nothing
        value2 = nothing
        endvalue=[]
    else
        value2 = mach.model        
        endvalue = Any[tree2(arg) for arg in mach.args]
    end
    keys = tuple(:operation,  :model, :args, :train_args)
    values = tuple(W.operation, value2,
                   Any[tree(arg) for arg in W.args],
                   endvalue)
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
    MLJ.reconstruct(tree)

Reconstruct a `Node` from its tree representation.

See also MLJ.tree

"""
function reconstruct(tree)
    if length(tree) == 1
        return first(tree)
    end
    values_ = values(tree)
    operation, model = values_[1], values_[2] 
    if model == nothing
        return node(operation, [reconstruct(arg) for arg in args(tree)]...)
    end
    mach = machine(model, [reconstruct(arg) for arg in train_args(tree)]...)
    return operation(mach, [reconstruct(arg) for arg in args(tree)]...)
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
network terminating at `W`, but replace any specified sources and
models `a1, a2, ...` of the original network by the specified targets
`b1, b2, ...`.

"""
function Base.replace(W::Node, pairs::Pair...) where N

    # Note: It is convenient to construct nodes of the new network as
    # values of a dictionary keyed on the nodes of the old
    # network. Additionally, there are dictionaries of models keyed on
    # old models and machines keyed on old machines. The node and
    # machine dictionaries must be built simultaneously.
    
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

# create a deep copy of the node N, with its sources stripped of
# content (data set to nothing):
function stripped_copy(N)
    sources = sources(N)
    X = sources[1].data
    y = sources[2].data
    sources[1].data = nothing
    sources[1].data = nothing
    
    Ncopy = deepcopy(N)
    
    # restore data:
    sources[1].data = X
    sources[2].data = y

    return Ncopy
end

# returns a fit method having node N as blueprint
function fit_method(N::Node)

    function fit(::Any, verbosity, X, y)
        yhat = MLJ.stripped_copy(N)
        X_, y_ = MLJ.sources(yhat)
        X_.data = X
        y_.data = y
        MLJ.reset!(yhat)
        fit!(yhat, verbosity=verbosity)
        cache = nothing
        report = nothing
        return yhat, cache, report
    end

    return fit
end
        
"""

   @composite NewCompositeModel(model1, model2, ...) <= N

Create a new stand-alone model type `NewCompositeModel` using the
learning network terminating at node `N` as a blueprint, equipping the
new type with field names `model1`, `model2`, ... . These fields point
to the component models in a deep copy of `N` that is created when an
instance of `NewCompositeModel` is first trained (ie, when `fit!` is
called on a machine binding the model to data). The counterparts of
these components in the original network `N` are the models
returned by `models(N)`, deep copies of which also serve as default
values for an automatically generated keywork constructor for
`NewCompositeModel`.

Return value: A new `NewCompositeModel` instance, with the default
field values detailed above. 

For details and examples refer to the "Learning Networks" section of
the documentation.

"""
macro composite(ex)
    modeltype_ex = ex.args[2].args[1]
    fieldname_exs = ex.args[2].args[2:end]
    N_ex = ex.args[3]
    composite_(__module__, modeltype_ex, fieldname_exs, N_ex)
    esc(quote
        $modeltype_ex()
        end)
end

function composite_(mod, modeltype_ex, fieldname_exs, N_ex)


    N = mod.eval(N_ex)
    N isa Node ||
        error("$(typeof(N)) bgiven where Node was expected. ")

    if models(N)[1] isa Supervised

        if MLJBase.is_probabilistic(typeof(models(N)[1]))
            subtype_ex = :ProbabilisticNetwork
        else
            subtype_ex = :DeterministicNetwork
        end

        # code defining the composite model struct and fit method:
        program1 = quote

            import MLJBase

            mutable struct $modeltype_ex <: MLJ.$subtype_ex
               $(fieldname_exs...)
            end

            MLJBase.fit(model::$modeltype_ex,
                        verbosity::Integer, X, y) =
                            MLJ.fit_method($N_ex)(model, verbosity, X, y)
        end

        program2 = quote
            MLJBase.@set_defaults($modeltype_ex,
                             MLJ.models(MLJ.stripped_copy(($N_ex))))
        end

        mod.eval(program1)   
        mod.eval(program2)
    else
        @warn "Did nothing"
    end

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

