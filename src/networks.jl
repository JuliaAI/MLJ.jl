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
    sources(N)

Return a list of all sources of a node `N` accessed by a call
`N()`. These are the sources of the acyclic directed graph terminating
at `N` of the associated learning network, if training input edges are
deleted.

See also: node, source

"""
sources(s::Source) = [s,]


## DEPENDENCY TAPES

# a tape is a vector of `NodalMachines` defined below, used to track dependencies
""" 
    merge!(tape1, tape2)

Incrementally appends to `tape1` all elements in `tape2`, excluding
any element previously added (or any element of `tape1` in its initial
state).

"""
function Base.merge!(tape1::Vector, tape2::Vector)
    for machine in tape2
        if !(machine in tape1)
            push!(tape1, machine)
        end
    end
    return tape1
end

# Note that `fit!` has already been defined for any  AbstractMachine in machines.jl

mutable struct NodalMachine{M<:Model} <: AbstractMachine{M}

    model::M
    previous_model::M
    fitresult
    cache
    args::Tuple{Vararg{AbstractNode}}
    report
    tape::Vector{NodalMachine}
    frozen::Bool
    rows                        # for remembering the rows used in last call to `fit!`
    state::Int                  # number of times fit! has been called on machine
    upstream_state::Vector{Int} # for remembering the upstream state in last call to `fit!`
    
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
#        machine.report = NamedTuple()

        # note: `get_tape(arg)` returns arg.tape where this makes
        # sense and an empty tape otherwise.  However, the complete
        # definition of `get_tape` must be postponed until
        # `Node` type is defined.

        # combine the tapes of all arguments to make a new tape:
        tape = get_tape(nothing) # returns blank tape 
        for arg in args
            merge!(tape, get_tape(arg))
        end
        machine.tape = tape
        machine.upstream_state = broadcast(m -> m.state, tape)

        return machine
    end
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


## NODES

struct Node{T<:Union{NodalMachine, Nothing}} <: AbstractNode

    operation             # that can be dispatched on a fit-result (eg, `predict`) or a static operation
    machine::T          # is `nothing` for static operations
    args::Tuple{Vararg{AbstractNode}}       # nodes where `operation` looks for its arguments
    sources::Vector{Source}
    tape::Vector{NodalMachine}    # for tracking dependencies

    function Node{T}(operation,
                     machine::T,
                     args::AbstractNode...) where {M<:Model, T<:Union{NodalMachine{M},Nothing}}

        # check the number of arguments:
        if machine == nothing
            length(args) > 0 || throw(error("`args` in `Node(::Function, args...)` must be non-empty. "))
        end

        sources_ = unique(vcat([sources(arg) for arg in args]...))
        length(sources_) == 1 ||
            @warn "Node with multiple non-training sources defined:\n$(sources_). "

        # get the machine's dependencies:
        tape = copy(get_tape(machine))

        # add the machine itself as a dependency:
        if machine != nothing
            merge!(tape, [machine, ])
        end

        # append the dependency tapes of all arguments:
        for arg in args
            merge!(tape, get_tape(arg))
        end

        return new{T}(operation, machine, args, sources_, tape)

    end
end

sources(X::Node) = X.sources

function is_stale(X::Node)
    (X.machine != nothing && is_stale(X.machine)) ||
        reduce(|, [is_stale(arg) for arg in X.args])
end

# to complete the definition of `NodalMachine` and `Node`
# constructors:
get_tape(::Any) = NodalMachine[]
get_tape(X::Node) = X.tape
get_tape(machine::NodalMachine) = machine.tape

# autodetect type parameter:
Node(operation, machine::M, args...) where M<:Union{NodalMachine, Nothing} =
    Node{M}(operation, machine, args...)

# constructor for static operations:
Node(operation, args::AbstractNode...) = Node(operation, nothing, args...)

# make nodes callable:
(y::Node)(; rows=:) = (y.operation)(y.machine, [arg(rows=rows) for arg in y.args]...)
function (y::Node)(Xnew)
    length(y.sources) == 1 ||
        error("Nodes with multiple non-training sources are not callable on new data. "*
              "Use sources(node) to inspect. ")
    return (y.operation)(y.machine, [arg(Xnew) for arg in y.args]...)
end

# Allow nodes to share the `selectrows(X, r)` syntax of concrete tabular data
# (needed for `fit(::AbstractMachine, ...)` in machines.jl):
MLJBase.selectrows(X::AbstractNode, r) = X(rows=r)

# and for the special case of static operations:
(y::Node{Nothing})(; rows=:) = (y.operation)([arg(rows=rows) for arg in y.args]...)
(y::Node{Nothing})(Xnew) = (y.operation)([arg(Xnew) for arg in y.args]...)

"""
    fit!(N::Node; rows=nothing, verbosity=1, force=false)

Call `fit!(mach)` on each machine `mach` upstream of `N`. 
"""
function fit!(y::Node; rows=nothing, verbosity=1, force=false)
    if rows == nothing
        rows = (:)
    end

    for mach in y.tape
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

See also: sources, node

"""
source(X) = Source(X) # here `X` is data

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

Shortcuts for `J = node(predcit, mach, X, y)`, etc. 

Calling a node is a recursive operation which terminates in the call
to a source node (or nodes). Calling nodes on *new* data `X` fails unless the
number of such nodes is one.  

See also: source, sources

"""
node = Node

# unless no arguments are `AbstractNode`s, `machine` creates a
# NodalTrainablaeModel, rather than a `Machine`:
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

