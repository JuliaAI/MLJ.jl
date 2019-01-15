## ABSTRACT NODES AND SOURCE NODES

abstract type AbstractNode <: MLJType end

struct Source{D} <: AbstractNode
    data::D      # training data
end

is_stale(s::Source) = false

# make source nodes callable:
function (s::Source)(; rows=:)
    if rows == (:)
        return s.data
    else
        return (s.data)[Rows, rows]
    end
end

(s::Source)(Xnew) = Xnew

get_sources(s::Source) = Set([s])


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

# TODO: replace linear tapes below with dependency trees to allow
# better scheduling of training learning networks data.

mutable struct NodalMachine{M<:Model} <: AbstractMachine{M}

    model::M
    previous_model::M
    fitresult
    cache
    args::Tuple{Vararg{AbstractNode}}
    report
    tape::Vector{NodalMachine}
    frozen::Bool
    rows # for remembering the rows used in last call to `fit!`
    
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
        machine.args = args
        machine.report = Dict{Symbol,Any}()

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

# fit method, general case (no coercion of arguments):
function fit!(machine::NodalMachine; rows=nothing, verbosity=1)

    if machine.frozen 
#        verbosity < 0 || @warn "$machine with model $(machine.model) "*
        verbosity < 0 || @warn "$machine "*
        "not trained as it is frozen."
        return machine
    end

    warning = clean!(machine.model)
    isempty(warning) || verbosity < 0 || @warn warning 
        
#    verbosity < 1 || @info "Training $machine whose model is $(machine.model)."
    verbosity < 1 || @info "Training $machine."

    if !isdefined(machine, :fitresult)
        if rows == nothing
            rows=(:) # error("An untrained NodalMachine requires rows to fit.")
        end
        args = [arg(rows=rows) for arg in machine.args]
        machine.fitresult, machine.cache, report =
            fit(machine.model, verbosity, args...)
        machine.rows = deepcopy(rows)
    else
        if rows == nothing # (ie rows not specified) update:
            args = [arg(rows=machine.rows) for arg in machine.args]
            machine.fitresult, machine.cache, report =
                update(machine.model, verbosity, machine.fitresult,
                       machine.cache, args...)
        else # retrain from scratch:
            args = [arg(rows=rows) for arg in machine.args]
            machine.fitresult, machine.cache, report =
                fit(machine.model, verbosity, args...)
            machine.rows = deepcopy(rows)
        end
    end

    machine.previous_model = deepcopy(machine.model)
    
    if report != nothing
        merge!(machine.report, report)
    end

    return machine

end

# fit method, supervised case (input data coerced):
function fit!(machine::NodalMachine{M}; rows=nothing, verbosity=1) where M<:Supervised

    if machine.frozen 
#        verbosity < 0 || @warn "$machine with model $(machine.model) "*
        verbosity < 0 || @warn "$machine "*
        "not trained as it is frozen."
        return machine
    end
        
#    verbosity < 1 || @info "Training $machine whose model is $(machine.model)."
    verbosity < 1 || @info "Training $machine."

    args = machine.args
    if !isdefined(machine, :fitresult)
        if rows == nothing
            rows=(:) # error("An untrained NodalMachine requires rows to fit.")
        end
        X = coerce(machine.model, args[1](rows=rows))
        ys = [arg(rows=rows) for arg in args[2:end]]
        machine.fitresult, machine.cache, report =
            fit(machine.model, verbosity, X, ys...)
        machine.rows = deepcopy(rows)
    else
        if rows == nothing # (ie rows not specified) update:
            X = coerce(machine.model, args[1](rows=machine.rows))
            ys = [arg(rows=machine.rows) for arg in args[2:end]]
            machine.fitresult, machine.cache, report =
                update(machine.model, verbosity, machine.fitresult,
                       machine.cache, X, ys...)
        else # retrain from scratch:
            X = coerce(machine.model, args[1](rows=rows))
            ys = [arg(rows=rows) for arg in args[2:end]]
            machine.fitresult, machine.cache, report =
                fit(machine.model, verbosity, X, ys...)
            machine.rows = deepcopy(rows)
        end
    end

    machine.previous_model = deepcopy(machine.model)
    
    if report != nothing
        merge!(machine.report, report)
    end

    return machine

end


## NODES

struct Node{T<:Union{NodalMachine, Nothing}} <: AbstractNode

    operation             # that can be dispatched on a fit-result (eg, `predict`) or a static operation
    machine::T          # is `nothing` for static operations
    args::Tuple{Vararg{AbstractNode}}       # nodes where `operation` looks for its arguments
    sources::Set{Source}
    tape::Vector{NodalMachine}    # for tracking dependencies

    function Node{T}(operation,
                     machine::T,
                     args::AbstractNode...) where {M<:Model, T<:Union{NodalMachine{M},Nothing}}

        # check the number of arguments:
        if machine == nothing
            length(args) > 0 || throw(error("`args` in `Node(::Function, args...)` must be non-empty. "))
        end

        sources = union([get_sources(arg) for arg in args]...)
        length(sources) == 1 || @warn "Node with multiple sources defined."

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

        return new{T}(operation, machine, args, sources, tape)

    end
end

# ... where
#get_depth(::Source) = 0
#get_depth(X::Node) = X.depth
get_sources(X::Node) = X.sources

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
    length(y.sources) == 1 || error("Nodes with multiple sources are not callable on new data. "*
                                    "The sources of the node called are $(y.sources)")
    return (y.operation)(y.machine, [arg(Xnew) for arg in y.args]...)
end

# and for the special case of static operations:
(y::Node{Nothing})(; rows=:) = (y.operation)([arg(rows=rows) for arg in y.args]...)
(y::Node{Nothing})(Xnew) = (y.operation)([arg(Xnew) for arg in y.args]...)

# if no `rows` specified, only retrain stale dependent
# NodalMachines (using whatever rows each one was last trained
# on):
function fit!(y::Node; rows=nothing, verbosity=1)
    if rows == nothing
        machines = filter(is_stale, y.tape)
    else
        machines = y.tape
    end
    for machine in machines
        fit!(machine; rows=rows, verbosity=verbosity-1)
    end
    return y
end

# allow arguments of `Nodes` and `NodalMachine`s to appear
# at REPL:
istoobig(d::Tuple{AbstractNode}) = length(d) > 10

# overload show method
function _recursive_show(stream::IO, X::AbstractNode)
    if X isa Source
        printstyled(IOContext(stream, :color=>true), MLJBase.handle(X), bold=true)
    else
        detail = (X.machine == nothing ? "(" : "($(MLJBase.handle(X.machine)), ")
        operation_name = typeof(X.operation).name.mt.name
        print(stream, operation_name, "(")
        if X.machine != nothing
            color = (X.machine.frozen ? :red : :green)
            printstyled(IOContext(stream, :color=>true), MLJBase.handle(X.machine),
                        bold=true)
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
    printstyled(IOContext(stream, :color=> true), str, bold=true)
    if !(X isa Source)
        print(stream, " = ")
        _recursive_show(stream, X)
    end
end
    
function Base.show(stream::IO, ::MIME"text/plain", machine::NodalMachine)
    id = objectid(machine) 
    description = string(typeof(machine).name.name)
    str = "$description @ $(MLJBase.handle(machine))"
    printstyled(IOContext(stream, :color=> true), str, bold=true)
    print(stream, " = ")
    print(stream, "machine($(machine.model), ")
    n_args = length(machine.args)
    counter = 1
    for arg in machine.args
        printstyled(IOContext(stream, :color=>true), MLJBase.handle(arg), bold=true)
        counter >= n_args || print(stream, ", ")
        counter += 1
    end
    print(stream, ")")
end

## SYNTACTIC SUGAR FOR LEARNING NETWORKS

source(X) = Source(X) # here `X` is data
# Node(X) = Source(X)   # here `X` is data
# Node(X::AbstractNode) = X 
node = Node

# unless no arguments are `AbstractNode`s, `machine` creates a
# NodalTrainablaeModel, rather than a `Machine`:
machine(model::Model, args::AbstractNode...) = NodalMachine(model, args...)
machine(model::Model, X, y::AbstractNode) = NodalMachine(model, source(X), y)
machine(model::Model, X::AbstractNode, y) = NodalMachine(model, X, source(y))

# aliases

# TODO: use macro to autogenerate these during model decleration/package glue-code:
# remove need for `Node` syntax in case of operations of main interest:
# predict(machine::NodalMachine, X::AbstractNode) = node(predict, machine, X)
# transform(machine::NodalMachine, X::AbstractNode) = node(transform, machine, X)
# inverse_transform(machine::NodalMachine, X::AbstractNode) = node(inverse_transform, machine, X)

matrix(X::AbstractNode) = node(matrix, X)

Base.log(v::Vector{<:Number}) = log.(v)
Base.exp(v::Vector{<:Number}) = exp.(v)
Base.log(X::AbstractNode) = node(log, X)
Base.exp(X::AbstractNode) = node(exp, X)


rms(y::AbstractNode, yhat::AbstractNode) = node(rms, y, yhat)
rms(y, yhat::AbstractNode) = node(rms, y, yhat)
rms(y::AbstractNode, yhat) = node(rms, y, yhat)

import Base.+
+(y1::AbstractNode, y2::AbstractNode) = node(+, y1, y2)
+(y1, y2::AbstractNode) = node(+, y1, y2)
+(y1::AbstractNode, y2) = node(+, y1, y2)


