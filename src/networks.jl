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

# a tape is a vector of `NodalTrainableModels` defined below, used to track dependencies
""" 
    merge!(tape1, tape2)

Incrementally appends to `tape1` all elements in `tape2`, excluding
any element previously added (or any element of `tape1` in its initial
state).

"""
function Base.merge!(tape1::Vector, tape2::Vector)
    for trainable_model in tape2
        if !(trainable_model in tape1)
            push!(tape1, trainable_model)
        end
    end
    return tape1
end

# TODO: replace linear tapes below with dependency trees to allow
# better scheduling of training learning networks data.

mutable struct NodalTrainableModel{M<:Model} <: AbstractTrainableModel{M}

    model::M
    previous_model::M
    fitresult
    cache
    args::Tuple{Vararg{AbstractNode}}
    report
    tape::Vector{NodalTrainableModel}
    frozen::Bool
    rows # for remembering the rows used in last call to `fit!`
    
    function NodalTrainableModel{M}(model::M, args::AbstractNode...) where M<:Model

        # check number of arguments for model subtypes:
        !(M <: Supervised) || length(args) == 2 ||
            throw(error("Wrong number of arguments. "*
                        "Use NodalTrainableModel(model, X, y) for supervised learner models."))
        !(M <: Unsupervised) || length(args) == 1 ||
            throw(error("Wrong number of arguments. "*
                        "Use NodalTrainableModel(model, X) for an unsupervised learner model."))
        
        trainable_model = new{M}(model)
        trainable_model.frozen = false
        trainable_model.args = args
        trainable_model.report = Dict{Symbol,Any}()

        # note: `get_tape(arg)` returns arg.tape where this makes
        # sense and an empty tape otherwise.  However, the complete
        # definition of `get_tape` must be postponed until
        # `Node` type is defined.

        # combine the tapes of all arguments to make a new tape:
        tape = get_tape(nothing) # returns blank tape 
        for arg in args
            merge!(tape, get_tape(arg))
        end
        trainable_model.tape = tape

        return trainable_model
    end
end

# automatically detect type parameter:
NodalTrainableModel(model::M, args...) where M<:Model = NodalTrainableModel{M}(model, args...)

# to turn fit-through fitting off and on:
function freeze!(trainable_model::NodalTrainableModel)
    trainable_model.frozen = true
end
function thaw!(trainable_model::NodalTrainableModel)
    trainable_model.frozen = false
end

function is_stale(trainable_model::NodalTrainableModel)
    !isdefined(trainable_model, :fitresult) ||
        trainable_model.model != trainable_model.previous_model ||
        reduce(|,[is_stale(arg) for arg in trainable_model.args])
end

# fit method, general case (no coercion of arguments):
function fit!(trainable_model::NodalTrainableModel; rows=nothing, verbosity=1)

    if trainable_model.frozen 
#        verbosity < 0 || @warn "$trainable_model with model $(trainable_model.model) "*
        verbosity < 0 || @warn "$trainable_model "*
        "not trained as it is frozen."
        return trainable_model
    end

    warning = clean!(trainable_model.model)
    isempty(warning) || verbosity < 0 || @warn warning 
        
#    verbosity < 1 || @info "Training $trainable_model whose model is $(trainable_model.model)."
    verbosity < 1 || @info "Training $trainable_model."

    if !isdefined(trainable_model, :fitresult)
        if rows == nothing
            rows=(:) # error("An untrained NodalTrainableModel requires rows to fit.")
        end
        args = [arg(rows=rows) for arg in trainable_model.args]
        trainable_model.fitresult, trainable_model.cache, report =
            fit(trainable_model.model, verbosity, args...)
        trainable_model.rows = deepcopy(rows)
    else
        if rows == nothing # (ie rows not specified) update:
            args = [arg(rows=trainable_model.rows) for arg in trainable_model.args]
            trainable_model.fitresult, trainable_model.cache, report =
                update(trainable_model.model, verbosity, trainable_model.fitresult,
                       trainable_model.cache, args...)
        else # retrain from scratch:
            args = [arg(rows=rows) for arg in trainable_model.args]
            trainable_model.fitresult, trainable_model.cache, report =
                fit(trainable_model.model, verbosity, args...)
            trainable_model.rows = deepcopy(rows)
        end
    end

    trainable_model.previous_model = deepcopy(trainable_model.model)
    
    if report != nothing
        merge!(trainable_model.report, report)
    end

    return trainable_model

end

# fit method, supervised case (input data coerced):
function fit!(trainable_model::NodalTrainableModel{M}; rows=nothing, verbosity=1) where M<:Supervised

    if trainable_model.frozen 
#        verbosity < 0 || @warn "$trainable_model with model $(trainable_model.model) "*
        verbosity < 0 || @warn "$trainable_model "*
        "not trained as it is frozen."
        return trainable_model
    end
        
#    verbosity < 1 || @info "Training $trainable_model whose model is $(trainable_model.model)."
    verbosity < 1 || @info "Training $trainable_model."

    args = trainable_model.args
    if !isdefined(trainable_model, :fitresult)
        if rows == nothing
            rows=(:) # error("An untrained NodalTrainableModel requires rows to fit.")
        end
        X = coerce(trainable_model.model, args[1](rows=rows))
        y = args[2](rows=rows)
        trainable_model.fitresult, trainable_model.cache, report =
            fit(trainable_model.model, verbosity, X, y)
        trainable_model.rows = deepcopy(rows)
    else
        if rows == nothing # (ie rows not specified) update:
            X = coerce(trainable_model.model, args[1](rows=trainable_model.rows))
            y = args[2](rows=trainable_model.rows)
            trainable_model.fitresult, trainable_model.cache, report =
                update(trainable_model.model, verbosity, trainable_model.fitresult,
                       trainable_model.cache, X, y)
        else # retrain from scratch:
            X = coerce(trainable_model.model, args[1](rows=rows))
            y = args[2](rows=rows)
            trainable_model.fitresult, trainable_model.cache, report =
                fit(trainable_model.model, verbosity, X, y)
            trainable_model.rows = deepcopy(rows)
        end
    end

    trainable_model.previous_model = deepcopy(trainable_model.model)
    
    if report != nothing
        merge!(trainable_model.report, report)
    end

    return trainable_model

end


## NODES

struct Node{T<:Union{NodalTrainableModel, Nothing}} <: AbstractNode

    operation             # that can be dispatched on a fit-result (eg, `predict`) or a static operation
    trainable::T          # is `nothing` for static operations
    args::Tuple{Vararg{AbstractNode}}       # nodes where `operation` looks for its arguments
    sources::Set{Source}
    tape::Vector{NodalTrainableModel}    # for tracking dependencies

    function Node{T}(operation,
                     trainable_model::T,
                     args::AbstractNode...) where {M<:Model, T<:Union{NodalTrainableModel{M},Nothing}}

        # check the number of arguments:
        if trainable_model == nothing
            length(args) > 0 || throw(error("`args` in `Node(::Function, args...)` must be non-empty. "))
        end

        sources = union([get_sources(arg) for arg in args]...)

        # get the trainable model's dependencies:
        tape = copy(get_tape(trainable_model))

        # add the trainable model itself as a dependency:
        if trainable_model != nothing
            merge!(tape, [trainable_model, ])
        end

        # append the dependency tapes of all arguments:
        for arg in args
            merge!(tape, get_tape(arg))
        end

        return new{T}(operation, trainable_model, args, sources, tape)

    end
end

# ... where
#get_depth(::Source) = 0
#get_depth(X::Node) = X.depth
get_sources(X::Node) = X.sources

function is_stale(X::Node)
    (X.trainable != nothing && is_stale(X.trainable)) ||
        reduce(|, [is_stale(arg) for arg in X.args])
end

# to complete the definition of `NodalTrainableModel` and `Node`
# constructors:
get_tape(::Any) = NodalTrainableModel[]
get_tape(X::Node) = X.tape
get_tape(trainable_model::NodalTrainableModel) = trainable_model.tape

# autodetect type parameter:
Node(operation, trainable_model::M, args...) where M<:Union{NodalTrainableModel, Nothing} =
    Node{M}(operation, trainable_model, args...)

# constructor for static operations:
Node(operation, args::AbstractNode...) = Node(operation, nothing, args...)

# make nodes callable:
(y::Node)(; rows=:) = (y.operation)(y.trainable, [arg(rows=rows) for arg in y.args]...)
function (y::Node)(Xnew)
    length(y.sources) == 1 || error("Nodes with multiple sources are not callable on new data. "*
                                    "Sources of node called = $(y.sources)")
    return (y.operation)(y.trainable, [arg(Xnew) for arg in y.args]...)
end

# and for the special case of static operations:
(y::Node{Nothing})(; rows=:) = (y.operation)([arg(rows=rows) for arg in y.args]...)
(y::Node{Nothing})(Xnew) = (y.operation)([arg(Xnew) for arg in y.args]...)

# if no `rows` specified, only retrain stale dependent
# NodalTrainableModels (using whatever rows each one was last trained
# on):
function fit!(y::Node; rows=nothing, verbosity=1)
    if rows == nothing
        trainable_models = filter(is_stale, y.tape)
    else
        trainable_models = y.tape
    end
    for trainable_model in trainable_models
        fit!(trainable_model; rows=rows, verbosity=verbosity-1)
    end
    return y
end

# allow arguments of `Nodes` and `NodalTrainableModel`s to appear
# at REPL:
istoobig(d::Tuple{AbstractNode}) = length(d) > 10

# overload show method
function _recursive_show(stream::IO, X::AbstractNode)
    if X isa Source
        printstyled(IOContext(stream, :color=>true), handle(X), bold=true)
    else
        detail = (X.trainable == nothing ? "(" : "($(handle(X.trainable)), ")
        operation_name = typeof(X.operation).name.mt.name
        print(stream, operation_name, "(")
        if X.trainable != nothing
            color = (X.trainable.frozen ? :red : :green)
            printstyled(IOContext(stream, :color=>true), handle(X.trainable),
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
    str = "$description @ $(handle(X))"
    printstyled(IOContext(stream, :color=> true), str, bold=true)
    if !(X isa Source)
        print(stream, " = ")
        _recursive_show(stream, X)
    end
end
    
function Base.show(stream::IO, ::MIME"text/plain", trainable_model::NodalTrainableModel)
    id = objectid(trainable_model) 
    description = string(typeof(trainable_model).name.name)
    str = "$description @ $(handle(trainable_model))"
    printstyled(IOContext(stream, :color=> true), str, bold=true)
    print(stream, " = ")
    print(stream, "trainable($(trainable_model.model), ")
    n_args = length(trainable_model.args)
    counter = 1
    for arg in trainable_model.args
        printstyled(IOContext(stream, :color=>true), handle(arg), bold=true)
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

# unless no arguments are `AbstractNode`s, `trainable` creates a
# NodalTrainablaeModel, rather than a `TrainableModel`:
trainable(model::Model, args::AbstractNode...) = NodalTrainableModel(model, args...)
trainable(model::Model, X, y::AbstractNode) = NodalTrainableModel(model, source(X), y)
trainable(model::Model, X::AbstractNode, y) = NodalTrainableModel(model, X, source(y))

# aliases

# TODO: use macro to autogenerate these during model decleration/package glue-code:
# remove need for `Node` syntax in case of operations of main interest:
# predict(trainable_model::NodalTrainableModel, X::AbstractNode) = node(predict, trainable_model, X)
# transform(trainable_model::NodalTrainableModel, X::AbstractNode) = node(transform, trainable_model, X)
# inverse_transform(trainable_model::NodalTrainableModel, X::AbstractNode) = node(inverse_transform, trainable_model, X)

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


## COMPOSITE MODELS

# When a supervised training network is wrapped as a stand-alone
# model plus interface, it is called a composite model.

# fall-back for updating supervised composite models:
function update(model::Supervised{Node}, verbosity, fitresult, cache, args...)
    fit!(fitresult; verbosity=verbosity)
    return fitresult, cache, nothing
end

