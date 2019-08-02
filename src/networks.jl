## ABSTRACT NODES AND SOURCE NODES

abstract type AbstractNode <: MLJType end

mutable struct Source <: AbstractNode
    data  # training data
end

is_stale(s::Source) = false

# make source nodes callable:
function (s::Source)(; rows=:)
    if rows == (:)
        return s.data
    else
        return selectrows(s.data, rows)
    end
end
(s::Source)(Xnew) = Xnew

"""
   rebind(s::Source, X)

Attach new data `X` to an existing source node `s`.

"""
function rebind!(s::Source, X)
    s.data = X
    return s
end


"""
    origins(N)

Return a list of all origins of a node `N` accessed by a call
`N()`. These are the source nodes of the acyclic directed graph
associated learning network terminating at `N` of the, if edges
corresponding to training arguments are excluded. A `Node` object
cannot be called on new data unles it has a unique origin.

Not to be confused with `sources(N)` which refers to the same graph
but without the training edge deletions.

See also: [`node`](@ref), [`source`](@ref).

"""
origins(s::Source) = [s,]

#  _merge!(nodes1, nodes2) incrementally appends to `nodes1` all
# elements in `nodes2`, excluding any element previously added (or any
# element of `nodes1` in its initial state).
function _merge!(nodes1, nodes2)
    for x in nodes2
        if !(x in nodes1)
            push!(nodes1, x)
        end
    end
    return nodes1
end

# Note that `fit!` has already been defined for any  AbstractMachine in machines.jl

mutable struct NodalMachine{M<:Model} <: AbstractMachine{M}

    model::M
    previous_model::M
    fitresult
    cache
    args::Tuple{Vararg{AbstractNode}}
    report
    frozen::Bool
    rows            # for remembering the rows used in last call to `fit!`
    state::Int      # number of times fit! has been called on machine
    upstream_state  # for remembering the upstream state in last call to `fit!`

    function NodalMachine{M}(model::M, args::AbstractNode...) where M<:Model

        # check number of arguments for model subtypes:
        !(M <: Supervised) || length(args) > 1 ||
            throw(error("Wrong number of arguments. "*
                        "You must provide target(s) for supervised models."))

        !(M <: Unsupervised) || length(args) == 1 ||
            throw(error("Wrong number of arguments. "*
                        "Use NodalMachine(model, X) for an unsupervised model."))

        machine = new{M}(model)
        machine.frozen = false
        machine.state = 0
        machine.args = args

        machine.upstream_state = Tuple([state(arg) for arg in args])

        return machine
    end
    NodalMachine{M}() where M<:Model = new{M}()
end

# automatically detect type parameter:
NodalMachine(model::M, args...) where M<:Model = NodalMachine{M}(model, args...)

# to turn fit-through fitting off and on:
function freeze!(machine::NodalMachine)
    machine.frozen = true
end
function thaw!(machine::NodalMachine)
    machine.frozen = false
end

function is_stale(machine::NodalMachine)
    !isdefined(machine, :fitresult) ||
        machine.model != machine.previous_model ||
        reduce(|,[is_stale(arg) for arg in machine.args])
end

state(machine::NodalMachine) = machine.state


## NODES

mutable struct Node{T<:Union{NodalMachine, Nothing}} <: AbstractNode

    operation  # that can be dispatched on a fit-result (eg, `predict`) or a static operation
    machine::T  # is `nothing` for static operations
    args::Tuple{Vararg{AbstractNode}}  # nodes where `operation` looks for its arguments
    origins::Vector{Source}
    nodes::Vector{AbstractNode}  # all upstream nodes, order consistent with DAG order (the node "tape")

    function Node{T}(operation,
                     machine::T,
                     args::AbstractNode...) where {M<:Model, T<:Union{NodalMachine{M},Nothing}}

        # check the number of arguments:
        if machine == nothing
            length(args) > 0 || throw(error("`args` in `Node(::Function, args...)` must be non-empty. "))
        end

        origins_ = unique(vcat([origins(arg) for arg in args]...))
        length(origins_) == 1 ||
            @warn "A node referencing multiple origins when called "*
        "has been defined:\n$(origins_). "

        # initialize the list of upstream nodes:
        nodes_ = AbstractNode[]

        # merge the lists from arguments:
        for arg in args
            _merge!(nodes_, nodes(arg))
        end

        # merge the lists from training arguments:
        if machine != nothing
            for arg in machine.args
                _merge!(nodes_, nodes(arg))
            end
        end

        return new{T}(operation, machine, args, origins_, nodes_)

    end
end

origins(X::Node) = X.origins
nodes(X::Node) = AbstractNode[X.nodes..., X]
nodes(S::Source) = AbstractNode[S, ]

function is_stale(X::Node)
    (X.machine != nothing && is_stale(X.machine)) ||
        reduce(|, [is_stale(arg) for arg in X.args])
end

state(s::MLJ.Source) = (state = 0, )
function state(W::MLJ.Node)
    mach = W.machine
    state_ =
        W.machine == nothing ? 0 : state(W.machine)
    if mach == nothing
        trainkeys=[]
        trainvalues=[]
    else
        trainkeys = [Symbol("train_arg", i) for i in eachindex(mach.args)]
        trainvalues = [state(arg) for arg in mach.args]
    end
    keys = tuple(:state,
                 [Symbol("arg", i) for i in eachindex(W.args)]...,
                 trainkeys...)
    values = tuple(state_,
                   [state(arg) for arg in W.args]...,
                   trainvalues...)
    return NamedTuple{keys}(values)
end

# autodetect type parameter:
Node(operation, machine::M, args...) where M<:Union{NodalMachine, Nothing} =
    Node{M}(operation, machine, args...)

# constructor for static operations:
Node(operation, args::AbstractNode...) = Node(operation, nothing, args...)

# make nodes callable:
(y::Node)(; rows=:) = (y.operation)(y.machine, [arg(rows=rows) for arg in y.args]...)
function (y::Node)(Xnew)
    length(y.origins) == 1 ||
        error("Nodes with multiple origins are not callable on new data. "*
              "Use origins(node) to inspect. ")
    return (y.operation)(y.machine, [arg(Xnew) for arg in y.args]...)
end

# Allow nodes to share the `selectrows(X, r)` syntax of concrete tabular data
# (needed for `fit(::AbstractMachine, ...)` in machines.jl):
MLJBase.selectrows(X::AbstractNode, r) = X(rows=r)

# and for the special case of static operations:
(y::Node{Nothing})(; rows=:) = (y.operation)([arg(rows=rows) for arg in y.args]...)
(y::Node{Nothing})(Xnew) = (y.operation)([arg(Xnew) for arg in y.args]...)

# Node <-> UUID mappings for efficient transfer of node identity between workers
const NODE_TO_UUID = IdDict{Any, UUID}()
const UUID_TO_NODE = Dict{UUID, Any}()
function _node_to_uuid(node)
    if haskey(NODE_TO_UUID, node)
        uuid = NODE_TO_UUID[node]
    else
        uuid = uuid4()
        NODE_TO_UUID[node] = uuid
        UUID_TO_NODE[uuid] = node
    end
    return uuid
end

"""
    fit!(N::Node; rows=nothing, verbosity=1, force=false)

Train the machines of all dynamic nodes in the learning network terminating at
`N` in an appropriate order.

"""
function fit!(y::Node; rows=nothing, verbosity=1, force=false, parallel=true) # FIXME: parallel=false should be the default
    if rows == nothing
        rows = (:)
    end

    if !parallel
        # get non-source nodes:
        nodes_ = filter(nodes(y)) do n
            n isa Node
        end

        # get machines to fit:
        machines = map(n -> n.machine, nodes_)
        machines = filter(unique(machines)) do mach
            mach != nothing
        end

        for mach in machines
            fit!(mach; rows=rows, verbosity=verbosity, force=force)
        end
    else
        @everywhere begin
            # HACK: Empty any pre-existing node/uuid mappings
            # FIXME: This is *super* inefficient for multiple `fit!` calls
            empty!(MLJ.NODE_TO_UUID)
            empty!(MLJ.UUID_TO_NODE)
        end

        # Register network (via UUIDs) on all workers
        # TODO: We can still use @everywhere here...
        rnodes = get_reduced_nodes(y)
        root = rnodes[NODE_TO_UUID[y]]
        for w in workers()
            fetch(@spawnat w MLJ.set_full_nodes(root, rnodes))
        end

        @everywhere begin
            println(length(keys(MLJ.NODE_TO_UUID)), " | ", length(keys(MLJ.UUID_TO_NODE)))
        end

        dag = construct_dag(y; rows=rows, verbosity=verbosity, force=force)
        @assert dag isa Thunk
        rnode, _ = collect(dag)
        DAGGER_DEBUG[] && @info "Finished full DAG"

        # sync final network back to master
        _sync!(y, rnode)
        DAGGER_DEBUG[] && @info "Synced back to master"
    end

    return y
end

# Convenience utility for constructing RemoteChannels for each worker
_make_rchans() = nothing#Dict{Int,RemoteChannel}(RemoteChannel(w) for w in workers())

function construct_dag(y::Node; rows=nothing, verbosity=1, force=false, mach_set=Set(), rchans=_make_rchans())
    DAGGER_DEBUG[] && printstyled("Construct DAG for $(typeof(y)), rows=$(repr(rows))\n"; color=:cyan)

    # get the DAGs of each arg
    arg_dags = Thunk[]
    append!(arg_dags, [construct_dag(arg; rows=rows, verbosity=verbosity, force=force, mach_set=mach_set, rchans=rchans) for arg in y.args])
    if y.machine !== nothing
        append!(arg_dags, [construct_dag(arg; rows=rows, verbosity=verbosity, force=force, mach_set=mach_set, rchans=rchans) for arg in y.machine.args])
    end

    # create the DAG for the node
    uniq_mach = false
    if y.machine !== nothing && !(y.machine in mach_set)
        uniq_mach = true
        push!(mach_set, y.machine)
    end
    uuid = _node_to_uuid(y)
    wdag = delayed((args...) -> begin
        my_y = UUID_TO_NODE[uuid]

        if DAGGER_DEBUG[]
            printstyled("In DAG for $(typeof(my_y))\n"; color=:green)
            printstyled(join(typeof.(args), ", "), '\n'; color=:magenta)
        end

        # Synchronize reduced arguments
        for arg in filter(a->(a isa Tuple && isreducednode(a[1])), collect(args))
            rnode, uuid = arg
            lnode = UUID_TO_NODE[uuid]
            _sync!(lnode, rnode)
            if DAGGER_DEBUG[]
                @info "Synced $arg for $my_y"
                if lnode.machine !== nothing && !isdefined(lnode.machine, :fitresult)
                    @warn "fitresult not defined!: $arg"
                end
            end
        end
        if uniq_mach
            if my_y.machine !== nothing
                DAGGER_DEBUG[] && printstyled("DAG: training $(typeof(my_y))\n"; color=:red)
                fit!(my_y.machine; rows=rows, verbosity=verbosity, force=force)
            else
                DAGGER_DEBUG[] && printstyled("DAG: not training static $(typeof(my_y))\n"; color=:red)
            end
        else
            DAGGER_DEBUG[] && printstyled("DAG: not training non-unique $(typeof(my_y))\n"; color=:red)
        end

        rnode, uuid = reduce_node(my_y)
        if uniq_mach && my_y.machine !== nothing
            if isdefined(my_y.machine, :fitresult)
                @assert isdefined(rnode.machine, :fitresult)
            end
        end
        return rnode, uuid
    end)(arg_dags...)
    # Force master to sync fitted node
    options = Dagger.Sch.ThunkOptions(1)
    mdag = delayed(rnode_uuid -> begin
        rnode, uuid = rnode_uuid #collect(c)
        lnode = UUID_TO_NODE[uuid]
        _sync!(lnode, rnode)
        return rnode_uuid
    end; options=options)(wdag)
    return mdag
end
function construct_dag(y::Source; rows=nothing, verbosity=1, force=false, mach_set=Set(), rchans=_make_rchans())
    DAGGER_DEBUG[] && printstyled("Construct DAG for Source, rows=$(repr(rows))\n"; color=:cyan)
    return delayed(identity)(y.data)
end

mutable struct ReducedNodalMachine{M} <: AbstractMachine{M}
    model::M
    previous_model::M
    fitresult
    cache
    args::Tuple{Vararg{UUID}}
    report
    frozen
    rows
    state
    upstream_state
    uuid::UUID
    ReducedNodalMachine{M}() where M = new{M}()
end
mutable struct ReducedNode{T<:Union{ReducedNodalMachine, Nothing}} <: AbstractNode
    operation
    machine::T
    args::Tuple{Vararg{UUID}}
    origins::Vector{UUID}
    nodes::Vector{UUID}
    uuid::UUID
end
struct ReducedSource <: AbstractNode
    data
    uuid::UUID
end
reduce_node(mach::NodalMachine) = (rmach = ReducedNodalMachine(mach); (rmach, rmach.uuid))
reduce_node(node::Node) = (rnode = ReducedNode(node); (rnode, rnode.uuid))
reduce_node(source::Source) = (rsource = ReducedSource(source); (rsource, rsource.uuid))
function ReducedNodalMachine(lmach::NodalMachine{M}, uuid::UUID=_node_to_uuid(lmach)) where M
    rmach = ReducedNodalMachine{M}()
    if isdefined(lmach, :model) rmach.model = lmach.model end
    if isdefined(lmach, :previous_model) rmach.previous_model = lmach.previous_model end
    if isdefined(lmach, :fitresult) rmach.fitresult = lmach.fitresult end
    if isdefined(lmach, :cache) rmach.cache = lmach.cache end
    rmach.args = last.(reduce_node.(lmach.args))
    if isdefined(lmach, :report) rmach.report = lmach.report end
    rmach.frozen = lmach.frozen
    if isdefined(lmach, :rows) rmach.rows = lmach.rows end
    rmach.state = lmach.state
    rmach.upstream_state = lmach.upstream_state
    rmach.uuid = uuid
    return rmach
end
function ReducedNode(node::Node, uuid::UUID=_node_to_uuid(node))
    if node.machine !== nothing
        mach = ReducedNodalMachine(node.machine)
    else
        mach = nothing
    end
    args = last.(reduce_node.(node.args))
    origins = last.(reduce_node.(node.origins))
    nodes = last.(reduce_node.(node.nodes))
    return ReducedNode(node.operation, mach, args, origins, nodes, uuid)
end
function ReducedSource(source::Source, uuid::UUID=_node_to_uuid(source))
    ReducedSource(source.data, uuid)
end

isreducednode(::Union{ReducedNodalMachine,ReducedNode,ReducedSource}) = true
isreducednode(x) = false

function get_reduced_nodes(lmach::NodalMachine, rnodes=Dict{UUID,Any}())
    rmach, uuid = reduce_node(lmach)
    rnodes[uuid] = rmach
    for arg in lmach.args
        get_reduced_nodes(arg, rnodes)
    end
    return rnodes
end
function get_reduced_nodes(lnode::Node, rnodes=Dict{UUID,Any}())
    rnode, uuid = reduce_node(lnode)
    if lnode.machine !== nothing
        get_reduced_nodes(lnode.machine, rnodes)
    end
    rnodes[uuid] = rnode
    for arg in lnode.args
        get_reduced_nodes(arg, rnodes)
    end
    return rnodes
end
function get_reduced_nodes(lsource::Source, rnodes=Dict{UUID,Any}())
    rsource, uuid = reduce_node(lsource)
    rnodes[uuid] = rsource
    return rnodes
end
set_full_nodes(root::ReducedNode, rnodes::Dict) = full_node(root, rnodes)
function full_node(rmach::ReducedNodalMachine{M}, rnodes::Dict) where M
    uuid = rmach.uuid
    haskey(UUID_TO_NODE, uuid) && return UUID_TO_NODE[uuid]
    if rmach !== nothing
        lmach = NodalMachine{M}()
        if isdefined(rmach, :model) lmach.model = rmach.model end
        if isdefined(rmach, :previous_model) lmach.previous_model = rmach.previous_model end
        if isdefined(rmach, :fitresult) lmach.fitresult = rmach.fitresult end
        if isdefined(rmach, :cache) lmach.cache = rmach.cache end
        lmach.args = ((_lookup_node(rnodes, uuid) for uuid in rmach.args)...,)
        if isdefined(rmach, :report) lmach.report = rmach.report end
        lmach.frozen = rmach.frozen
        if isdefined(rmach, :rows) lmach.rows = rmach.rows end
        lmach.state = rmach.state
        lmach.upstream_state = rmach.upstream_state

        #=
        lmach.model = rmach.model
        lmach.previous_model = rmach.previous_model
        lmach.fitresult = rmach.fitresult
        lmach.cache = rmach.cache
        lmach.args = ((_lookup_node(rnodes, uuid) for uuid in rmach.args)...,)
        lmach.report = rmach.report
        lmach.frozen = rmach.frozen
        lmach.rows = rmach.rows
        lmach.state = rmach.state
        lmach.upstream_state = rmach.upstream_state
        =#
    else
        lmach = nothing
    end
    NODE_TO_UUID[lmach] = uuid
    UUID_TO_NODE[uuid] = lmach
    return lmach
end
full_node(rmach::Nothing, rnodes) = nothing
function full_node(rnode::ReducedNode, rnodes::Dict)
    uuid = rnode.uuid
    haskey(UUID_TO_NODE, uuid) && return UUID_TO_NODE[uuid]
    args = [_lookup_node(rnodes, arg) for arg in rnode.args]
    origins = [_lookup_node(rnodes, origin) for origin in rnode.origins]
    nodes = [_lookup_node(rnodes, node) for node in rnode.nodes]
    lnode = Node(rnode.operation, full_node(rnode.machine, rnodes), args...) #, origins, nodes)
    NODE_TO_UUID[lnode] = uuid
    UUID_TO_NODE[uuid] = lnode
    return lnode
end
function full_node(rsource::ReducedSource, rnodes::Dict)
    uuid = rsource.uuid
    uuid in keys(UUID_TO_NODE) && return UUID_TO_NODE[uuid]
    lsource = Source(rsource.data)
    NODE_TO_UUID[lsource] = uuid
    UUID_TO_NODE[uuid] = lsource
    return lsource
end
function _lookup_node(rnodes::Dict, uuid::UUID)
    if haskey(UUID_TO_NODE, uuid)
        return UUID_TO_NODE[uuid]
    else
        # TODO: Make this an assertion
        if !haskey(rnodes, uuid)
            @warn "UUID $uuid not found in rnodes"
            @show rnodes
            throw(ErrorException("UUID lookup failure"))
        end
        return full_node(rnodes[uuid], rnodes)
    end
end
function _sync!(lnode::Node, rnode::ReducedNode)
    lnode.operation = rnode.operation
    if rnode.machine !== nothing
        lmach = lnode.machine
        rmach = rnode.machine

        lmach.model = rmach.model
        lmach.previous_model = rmach.previous_model
        lmach.fitresult = rmach.fitresult
        lmach.cache = rmach.cache
        lmach.args = ((UUID_TO_NODE[uuid] for uuid in rmach.args)...,)
        lmach.report = rmach.report
        lmach.frozen = rmach.frozen
        lmach.rows = rmach.rows
        lmach.state = rmach.state
        lmach.upstream_state = rmach.upstream_state
    end
    lnode.args = ((UUID_TO_NODE[arg] for arg in rnode.args)...,)
    lnode.origins = [UUID_TO_NODE[origin] for origin in rnode.origins]
    lnode.nodes = [UUID_TO_NODE[node] for node in rnode.nodes]
end
_sync!(lsource::Source, rsource::ReducedSource) = (lsource.data = rsource.data;)

# allow arguments of `Nodes` and `NodalMachine`s to appear
# at REPL:
istoobig(d::Tuple{AbstractNode}) = length(d) > 10

# overload show method



function _recursive_show(stream::IO, X::AbstractNode)
    if X isa Source
        printstyled(IOContext(stream, :color=>MLJBase.SHOW_COLOR), MLJBase.handle(X), color=:blue)
    else
        detail = (X.machine == nothing ? "(" : "($(MLJBase.handle(X.machine)), ")
        operation_name = typeof(X.operation).name.mt.name
        print(stream, operation_name, "(")
        if X.machine != nothing
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

function Base.show(stream::IO, ::MIME"text/plain", X::AbstractNode)
    id = objectid(X)
    description = string(typeof(X).name.name)
    str = "$description @ $(MLJBase.handle(X))"
    printstyled(IOContext(stream, :color=>MLJBase.SHOW_COLOR), str, color=:blue)
    if !(X isa Source)
        print(stream, " = ")
        _recursive_show(stream, X)
    end
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
    Xs = source(X)

Defines a `Source` object out of data `X`. The data can be a vector,
categorical vector, or table. The calling behaviour of a source node is this:

    Xs() = X
    Xs(rows=r) = selectrows(X, r)  # eg, X[r,:] for a DataFrame
    Xs(Xnew) = Xnew

See also: [`origins`](@ref), [`node`](@ref).

"""
source(X) = Source(X) # here `X` is data
source(X::Source) = X

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
const node = Node

# unless no arguments are `AbstractNode`s, `machine` creates a
# NodalTrainableModel, rather than a `Machine`:
machine(model::Model, args::AbstractNode...) = NodalMachine(model, args...)
machine(model::Model, X, y::AbstractNode) = NodalMachine(model, source(X), y)
machine(model::Model, X::AbstractNode, y) = NodalMachine(model, X, source(y))

MLJBase.matrix(X::AbstractNode) = node(MLJBase.matrix, X)

Base.log(X::AbstractNode) = node(v->log.(v), X)
Base.exp(X::AbstractNode) = node(v->exp.(v), X)

import Base.+
+(y1::AbstractNode, y2::AbstractNode) = node(+, y1, y2)
+(y1, y2::AbstractNode) = node(+, y1, y2)
+(y1::AbstractNode, y2) = node(+, y1, y2)

import Base.*
*(lambda::Real, y::AbstractNode) = node(y->lambda*y, y)
