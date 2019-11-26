## ABSTRACT NODES AND SOURCE NODES

abstract type AbstractNode <: MLJType end

# K is :target, :input, :weight or :unknown
mutable struct Source{K} <: AbstractNode
    data  # training data
end

"""
    Xs = source(X)
    ys = source(y, kind=:target)
    ws = source(w, kind=:weight)

Defines, respectively, learning network `Source` objects for wrapping
some input data `X` (`kind=:input`), some target data `y`, or some
sample weights `w`.  The values of each variable `X, y, w` can be
anything, even `nothing`, if the network is for exporting as a
stand-alone model only. For training and testing the unexported network,
appropriate vectors, tables, or other data containers are expected.

    Xs = source()
    ys = source(kind=:target)
    ws = source(kind=:weight)

Define source nodes wrapping `nothing` instead of concrete data. Such
definitions suffice if a learning network is to be exported without
testing.

The calling behaviour of a `Source` object is this:

    Xs() = X
    Xs(rows=r) = selectrows(X, r)  # eg, X[r,:] for a DataFrame
    Xs(Xnew) = Xnew

See also: [`@from_network`](@ref], [`sources`](@ref),
[`origins`](@ref), [`node`](@ref).

"""
function source(X; kind=:input)
    kind in [:input, :target, :weights] ||
        @warn "`Source` kind is neither :input, :target, or :weights. "
    return Source{kind}(X)
end

source(X::Source; args...) = X
source(; args...) = source(nothing; args...)

is_stale(s::Source) = false

# make source nodes callable:
function (s::Source)(; rows=:)
    rows == (:) && return s.data
    return selectrows(s.data, rows)
end
(s::Source)(Xnew) = Xnew

"""
    rebind!(s)

Attach new data `X` to an existing source node `s`.
"""
function rebind!(s::Source, X)
    s.data = X
    return s
end


"""
$SIGNATURES

Return a list of all origins of a node `N` accessed by a call `N()`.
These are the source nodes of the acyclic directed graph (DAG)
associated with the learning network terminating at `N`, if edges
corresponding to training arguments are excluded. A `Node` object
cannot be called on new data unless it has a unique origin.

Not to be confused with `sources(N)` which refers to the same graph
but without the training edge deletions.

See also: [`node`](@ref), [`source`](@ref).
"""
origins(s::Source) = [s,]


"""
$SIGNATURES

Incrementally append to `nodes1` all elements in `nodes2`, excluding any elements
previously added (or any element of `nodes1` in its initial state).
"""
function _merge!(nodes1, nodes2)
    for x in nodes2
        if !(x in nodes1)
            push!(nodes1, x)
        end
    end
    return nodes1
end

# Note that `fit!` has already been defined for any  AbstractMachine in machines.jl

"""
$TYPEDEF

A NodalMachine wraps a model as part of a learning network.
"""
mutable struct NodalMachine{M<:Model} <: AbstractMachine{M}

    model::M
    previous_model::M # for remembering the model used in last call to `fit!`
    fitresult
    cache
    args::Tuple{Vararg{AbstractNode}}
    report
    frozen::Bool
    previous_rows   # for remembering the rows used in last call to `fit!`
    state::Int      # number of times fit! has been called on machine
    upstream_state  # for remembering the upstream state in last call to `fit!`

    function NodalMachine{M}(model::M, args::AbstractNode...) where M<:Model

        # check number of arguments for model subtypes:
        !(M <: Supervised) || length(args) > 1 ||
            throw(error("Wrong number of arguments. " *
                        "You must provide target(s) for supervised models."))

        if M <: Unsupervised && !(M <: Static)
            length(args) == 1 ||
                throw(error("Wrong number of arguments. " *
                            "Use NodalMachine(model, X) for an "*
                            "unsupervised model."))
        end

        machine = new{M}(model)
        machine.frozen = false
        machine.state = 0
        machine.args = args

        machine.upstream_state = Tuple([state(arg) for arg in args])

        return machine
    end
end

# automatically detect type parameter:
NodalMachine(model::M, args...) where M<:Model = NodalMachine{M}(model, args...)

"""
$SIGNATURES

Freeze the machine `machine` so that it will never be retrained (unless it is thawed).
See also [`thaw!`](@ref).
"""
function freeze!(machine::NodalMachine)
    machine.frozen = true
end

"""
$SIGNATURES

Unfreeze the machine `machine` so that it can be retrained.
See also [`freeze!`](@ref).
"""
function thaw!(machine::NodalMachine)
    machine.frozen = false
end

"""
$SIGNATURES

Check if a machine is stale.
"""
function is_stale(machine::NodalMachine)
    !isdefined(machine, :fitresult) ||
        machine.model != machine.previous_model ||
        reduce(|,[is_stale(arg) for arg in machine.args])
end

"""
$SIGNATURES

Return the state of a machine.
"""
state(machine::NodalMachine) = machine.state


## NODES

struct Node{T<:Union{NodalMachine, Nothing}} <: AbstractNode

    operation   # that can be dispatched on a fit-result (eg, `predict`) or a static operation
    machine::T  # is `nothing` for static operations
    args::Tuple{Vararg{AbstractNode}}  # nodes where `operation` looks for its arguments
    origins::Vector{Source}
    nodes::Vector{AbstractNode}  # all upstream nodes, order consistent with DAG order (the node "tape")

    function Node{T}(operation,
                     machine::T,
                     args::AbstractNode...) where {M<:Model, T<:Union{NodalMachine{M},Nothing}}

        # check the number of arguments:
        if machine === nothing && isempty(args)
            throw(error("`args` in `Node(::Function, args...)` must be non-empty. "))
        end

        origins_ = unique(vcat([origins(arg) for arg in args]...))
        # length(origins_) == 1 ||
        #     @warn "A node referencing multiple origins when called " *
        #           "has been defined:\n$(origins_). "

        # initialize the list of upstream nodes:
        nodes_ = AbstractNode[]

        # merge the lists from arguments:
        for arg in args
            _merge!(nodes_, nodes(arg))
        end

        # merge the lists from training arguments:
        if machine !== nothing
            for arg in machine.args
                _merge!(nodes_, nodes(arg))
            end
        end

        return new{T}(operation, machine, args, origins_, nodes_)
    end
end

"""
$SIGNATURES

Access the origins (source nodes) of a given node.
"""
origins(X::Node) = X.origins

"""
$SIGNATURES

Return all nodes upstream of a node, including the node, in order consistent with the DAG.
"""
nodes(X::Node) = AbstractNode[X.nodes..., X]
nodes(S::Source) = AbstractNode[S, ]

"""
$SIGNATURES

Check if a node is stale.
"""
function is_stale(X::Node)
    (X.machine !== nothing && is_stale(X.machine)) ||
        reduce(|, [is_stale(arg) for arg in X.args])
end

state(s::MLJ.Source) = (state = 0, )

"""
$SIGNATURES

Return the state of a node.
"""
function state(W::MLJ.Node)
    mach = W.machine
    state_ = W.machine === nothing ? 0 : state(W.machine)
    if mach === nothing
        trainkeys   = []
        trainvalues = []
    else
        trainkeys   = (Symbol("train_arg", i) for i in eachindex(mach.args))
        trainvalues = (state(arg) for arg in mach.args)
    end
    keys = tuple(:state,
                 (Symbol("arg", i) for i in eachindex(W.args))...,
                 trainkeys...)
    values = tuple(state_,
                   (state(arg) for arg in W.args)...,
                   trainvalues...)
    return NamedTuple{keys}(values)
end

# autodetect type parameter:
Node(operation, machine::M, args...) where M <: Union{NodalMachine,Nothing} =
    Node{M}(operation, machine, args...)

# constructor for static operations:
Node(operation, args::AbstractNode...) = Node(operation, nothing, args...)

# make nodes callable:
(y::Node)(; rows=:) =
    (y.operation)(y.machine, [arg(rows=rows) for arg in y.args]...)
function (y::Node)(Xnew)
    length(y.origins) == 1 ||
        error("Node $y has multiple origins and cannot be called "*
              "on new data. ")
    return (y.operation)(y.machine, [arg(Xnew) for arg in y.args]...)
end

# and for the special case of static operations:
(y::Node{Nothing})(; rows=:) =
    (y.operation)([arg(rows=rows) for arg in y.args]...)
function (y::Node{Nothing})(Xnew)
    length(y.origins) == 1 ||
        error("Node $y has multiple origins and cannot be called "*
              "on new data. ")
    return (y.operation)([arg(Xnew) for arg in y.args]...)
end

"""
    fit!(N::Node; rows=nothing, verbosity::Int=1, force::Bool=false)

Train all machines in the learning network terminating at node `N`, in an
appropriate order. These machines are those returned by `machines(N)`.

"""
function fit!(y::Node; rows=nothing, verbosity::Int=1,
force::Bool=false) if rows === nothing rows = (:) end

    # get non-source nodes:
    nodes_ = filter(nodes(y)) do n
        n isa Node
    end

    # get machines to fit:
    machines = map(n -> n.machine, nodes_)
    machines = filter(unique(machines)) do mach
        mach !== nothing
    end

    for mach in machines
        fit!(mach; rows=rows, verbosity=verbosity, force=force)
    end

    return y
end

# allow arguments of `Nodes` and `NodalMachine`s to appear
# at REPL:
istoobig(d::Tuple{AbstractNode}) = length(d) > 10

# overload show method

function _recursive_show(stream::IO, X::AbstractNode)
    if X isa Source
        printstyled(IOContext(stream, :color=>MLJBase.SHOW_COLOR), MLJBase.handle(X), color=:blue)
    else
        detail = (X.machine === nothing ? "(" : "($(MLJBase.handle(X.machine)), ")
        operation_name = typeof(X.operation).name.mt.name
        print(stream, operation_name, "(")
        if X.machine !== nothing
            color = (X.machine.frozen ? :red : :green)
            printstyled(IOContext(stream, :color=>MLJBase.SHOW_COLOR), MLJBase.handle(X.machine),
                        bold=MLJBase.SHOW_COLOR)
            print(stream, ", ")
        end
        n_args = length(X.args)
        counter = 1
        for arg in X.args
            _recursive_show(stream, arg)
            counter >= n_args || print(stream, ", ")
            counter += 1
        end
        print(stream, ")")
    end
end

function Base.show(stream::IO, ::MIME"text/plain", X::Node)
    id = objectid(X)
    description = string(typeof(X).name.name)
    str = "$description @ $(MLJBase.handle(X))"
    printstyled(IOContext(stream, :color=>MLJBase.SHOW_COLOR), str, color=:blue)
#    if !(X isa Source)
        print(stream, " = ")
        _recursive_show(stream, X)
#    end
end

function Base.show(stream::IO, ::MIME"text/plain", machine::NodalMachine)
    id = objectid(machine)
    description = string(typeof(machine).name.name)
    str = "$description @ $(MLJBase.handle(machine))"
    printstyled(IOContext(stream, :color=>MLJBase.SHOW_COLOR), str, bold=MLJBase.SHOW_COLOR)
    print(stream, " = ")
    print(stream, "machine($(machine.model), ")
    n_args = length(machine.args)
    counter = 1
    for arg in machine.args
        printstyled(IOContext(stream, :color=>MLJBase.SHOW_COLOR), MLJBase.handle(arg), bold=MLJBase.SHOW_COLOR)
        counter >= n_args || print(stream, ", ")
        counter += 1
    end
    print(stream, ")")
end

## SYNTACTIC SUGAR FOR LEARNING NETWORKS

"""
    N = node(f::Function, args...)

Defines a `Node` object `N` wrapping a static operation `f` and arguments
`args`. Each of the `n` elements of `args` must be a `Node` or `Source`
object. The node `N` has the following calling behaviour:

    N() = f(args[1](), args[2](), ..., args[n]())
    N(rows=r) = f(args[1](rows=r), args[2](rows=r), ..., args[n](rows=r))
    N(X) = f(args[1](X), args[2](X), ..., args[n](X))

    J = node(f, mach::NodalMachine, args...)

Defines a dynamic `Node` object `J` wrapping a dynamic operation `f`
(`predict`, `predict_mean`, `transform`, etc), a nodal machine `mach` and
arguments `args`. Its calling behaviour, which depends on the outcome of
training `mach` (and, implicitly, on training outcomes affecting its
arguments) is this:

    J() = f(mach, args[1](), args[2](), ..., args[n]())
    J(rows=r) = f(mach, args[1](rows=r), args[2](rows=r), ..., args[n](rows=r))
    J(X) = f(mach, args[1](X), args[2](X), ..., args[n](X))

Generally `n=1` or `n=2` in this latter case.

    predict(mach, X::AbsractNode, y::AbstractNode)
    predict_mean(mach, X::AbstractNode, y::AbstractNode)
    predict_median(mach, X::AbstractNode, y::AbstractNode)
    predict_mode(mach, X::AbstractNode, y::AbstractNode)
    transform(mach, X::AbstractNode)
    inverse_transform(mach, X::AbstractNode)

Shortcuts for `J = node(predict, mach, X, y)`, etc.

Calling a node is a recursive operation which terminates in the call
to a source node (or nodes). Calling nodes on *new* data `X` fails unless the
number of such nodes is one.

See also: [`source`](@ref), [`origins`](@ref).

"""
node = Node

# unless no arguments are `AbstractNode`s, `machine` creates a
# NodalTrainableModel, rather than a `Machine`:
machine(model::Model, args::AbstractNode...) = NodalMachine(model, args...)
machine(model::Model, X, y::AbstractNode) = NodalMachine(model, source(X), y)
machine(model::Model, X::AbstractNode, y) = NodalMachine(model, X, source(y))

MLJBase.matrix(X::AbstractNode) = node(MLJBase.matrix, X)
MLJBase.table(X::AbstractNode) = node(MLJBase.table, X)
Base.vcat(args::AbstractNode...) = node(vcat, args...)
Base.hcat(args::AbstractNode...) = node(hcat, args...)

Statistics.mean(X::AbstractNode) = node(v->mean.(v), X)
Statistics.median(X::AbstractNode) = node(v->median.(v), X)
import StatsBase.mode
StatsBase.mode(X::AbstractNode) = node(v->mode.(v), X)

Base.log(X::AbstractNode) = node(v->log.(v), X)
Base.exp(X::AbstractNode) = node(v->exp.(v), X)

import Base.+
+(y1::AbstractNode, y2::AbstractNode) = node(+, y1, y2)
+(y1, y2::AbstractNode) = node(+, y1, y2)
+(y1::AbstractNode, y2) = node(+, y1, y2)

import Base.*
*(lambda::Real, y::AbstractNode) = node(y->lambda*y, y)

"""
    selectcols(X::AbstractNode, c)

Returns `Node` object `N` such that `N() = selectcols(X(), c)`.

"""
MLJBase.selectcols(X::AbstractNode, r) = node(XX->selectcols(XX, r),
X)

"""
    selectrows(X::AbstractNode, r)

Returns a `Node` object `N` such that `N() = selectrows(X(), r)` (and
`N(rows=s) = selectrows(X(rows=s), r)`).

"""
MLJBase.selectrows(X::AbstractNode, r) = node(XX->selectrows(XX, r),
X)


## INSPECTING LEARNING NETWORKS

"""
$SIGNATURES

Return a description of the tree defined by the learning network
terminating at a given node.
"""
function tree(W::MLJ.Node)
    mach = W.machine
    if mach === nothing
        value2 = nothing
        endkeys = []
        endvalues = []
    else
        value2 = mach.model
        endkeys = (Symbol("train_arg", i) for i in eachindex(mach.args))
        endvalues = (tree(arg) for arg in mach.args)
    end
    keys = tuple(:operation,  :model,
                 (Symbol("arg", i) for i in eachindex(W.args))...,
                 endkeys...)
    values = tuple(W.operation, value2,
                   (tree(arg) for arg in W.args)...,
                   endvalues...)
    return NamedTuple{keys}(values)
end
tree(s::MLJ.Source) = (source = s,)

"""
$SIGNATURES

Return a vector of the top level args of the tree associated with a node.
If `train=true`, return the `train_args`.
"""
function args(tree; train=false)
    keys_ = filter(keys(tree) |> collect) do key
        match(Regex("^$("train_"^train)arg[0-9]*"), string(key)) !== nothing
    end
    return [getproperty(tree, key) for key in keys_]
end

"""
$SIGNATURES

A vector of all models referenced by a node, each model appearing exactly once.
"""
function MLJModels.models(W::MLJ.AbstractNode)
    models_ = filter(flat_values(tree(W)) |> collect) do model
        model isa MLJ.Model
    end
    return unique(models_)
end

"""
    sources(W::AbstractNode; kind=:any)

A vector of all sources referenced by calls `N()` and `fit!(N)`. These
are the sources of the directed acyclic graph associated with the
learning network terminating at `N`. The return value can be
restricted further by specifying `kind=:input`, `kind=:target`,
`kind=:weight`, etc.

Not to be confused with `origins(N)` which refers to the same graph with edges
corresponding to training arguments deleted.

See also: [`origins`](@ref), [`source`](@ref).
"""
function sources(W::MLJ.AbstractNode; kind=:any)
    if kind == :any
        sources_ = filter(MLJ.flat_values(tree(W)) |> collect) do value
            value isa MLJ.Source
        end
    else
        sources_ = filter(MLJ.flat_values(tree(W)) |> collect) do value
            value isa MLJ.Source{kind}
        end
    end
    return unique(sources_)
end


"""
$SIGNATURES

List all machines in the learning network terminating at a given node.
"""
function machines(W::MLJ.Node)
    if W.machine === nothing
        return vcat((machines(arg) for arg in W.args)...) |> unique
    else
        return vcat(Any[W.machine, ],
                    (machines(arg) for arg in W.args)...,
                    (machines(arg) for arg in W.machine.args)...) |> unique
    end
end
machines(W::MLJ.Source) = Any[]


## MANIPULATING LEARNING NETWORKS

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
