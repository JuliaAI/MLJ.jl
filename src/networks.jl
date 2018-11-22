abstract type AbstractNode <: MLJType end

# a tape is a vector of `NodalTrainableModels` defined below, used to track dependencies
""" 
    merge!(tape1, tape2)

Incrementally appends to `tape1` all elements in `tape2`, excluding
any element previously added (or any element of `tape1` in its initial
state).

"""
function Base.merge!(tape1::Vector, tape2::Vector)
    for trainable in tape2
        if !(trainable in tape1)
            push!(tape1, trainable)
        end
    end
    return tape1
end

# TODO: replace linear tapes below with dependency trees to allow
# better scheduling of training learning networks data.

mutable struct NodalTrainableModel{B<:Model} <: MLJType

    model::B
    previous_model::B
    fitresult
    cache
    args::Tuple{Vararg{AbstractNode}}
    report
    tape::Vector{NodalTrainableModel}
    frozen::Bool
    rows # for remembering the rows used in last call to `fit!`
    
    function NodalTrainableModel{B}(model::B, args::AbstractNode...) where B<:Model

        # check number of arguments for model subtypes:
        !(B <: Supervised) || length(args) == 2 ||
            throw(error("Wrong number of arguments. "*
                        "Use NodalTrainableModel(model, X, y) for supervised learner models."))
        !(B <: Unsupervised) || length(args) == 1 ||
            throw(error("Wrong number of arguments. "*
                        "Use NodalTrainableModel(model, X) for an unsupervised learner model."))
        
        trainable = new{B}(model)
        trainable.frozen = false
        trainable.args = args
        trainable.report = Dict{Symbol,Any}()

        # note: `get_tape(arg)` returns arg.tape where this makes
        # sense and an empty tape otherwise.  However, the complete
        # definition of `get_tape` must be postponed until
        # `Node` type is defined.

        # combine the tapes of all arguments to make a new tape:
        tape = get_tape(nothing) # returns blank tape 
        for arg in args
            merge!(tape, get_tape(arg))
        end
        trainable.tape = tape

        return trainable
    end
end

# automatically detect type parameter:
NodalTrainableModel(model::B, args...) where B<:Model = NodalTrainableModel{B}(model, args...)

# to turn fit-through fitting off and on:
function freeze!(trainable::NodalTrainableModel)
    trainable.frozen = true
end
function thaw!(trainable::NodalTrainableModel)
    trainable.frozen = false
end

function is_stale(trainable::NodalTrainableModel)
    !isdefined(trainable, :fitresult) ||
        trainable.model != trainable.previous_model ||
        reduce(|,[is_stale(arg) for arg in trainable.args])
end

# fit method:
function fit!(trainable::NodalTrainableModel, rows=nothing; verbosity=1)

    if trainable.frozen 
        verbosity < 0 || @warn "$trainable with model $(trainable.model) "*
        "not trained as it is frozen."
        return trainable
    end
        
    verbosity < 1 || @info "Training $trainable whose model is $(trainable.model)."

    # call up the data at relevant source nodes for training:

    if !isdefined(trainable, :fitresult)
        rows != nothing || error("An untrained NodalTrainableModel requires rows to fit.")
        args = [arg()[Rows, rows] for arg in trainable.args]
        trainable.fitresult, trainable.cache, report =
            fit(trainable.model, verbosity, args...)
        trainable.rows = deepcopy(rows)
    else
        if rows == nothing # (ie rows not specified) update:
            args = [arg()[Rows, trainable.rows] for arg in trainable.args]
            trainable.fitresult, trainable.cache, report =
                update(trainable.model, verbosity, trainable.fitresult,
                       trainable.cache, args...)
        else # retrain from scratch:
            args = [arg()[Rows, rows] for arg in trainable.args]
            trainable.fitresult, trainable.cache, report =
                fit(trainable.model, verbosity, args...)
            trainable.rows = deepcopy(rows)
        end
    end

    trainable.previous_model = deepcopy(trainable.model)
    
    if report != nothing
        merge!(trainable.report, report)
    end

    return trainable

end

# TODO: avoid repeated code below using macro and possibly move out of
# networks.jl

# predict method for trainable learner models (X data):
function predict(trainable::NodalTrainableModel, X) 
    if isdefined(trainable, :fitresult)
        return predict(trainable.model, trainable.fitresult, X)
    else
        throw(error("$trainable with model $(trainable.model) is not trained and so cannot predict."))
    end
end

# a transform method for trainable transformer models (X data):
function transform(trainable::NodalTrainableModel, X)
    if isdefined(trainable, :fitresult)
        return transform(trainable.model, trainable.fitresult, X)
    else
        throw(error("$trainable with model $(trainable.model) is not trained and so cannot transform."))
    end
end

# an inverse-transform method for trainable transformer models (X data):
function inverse_transform(trainable::NodalTrainableModel, X)
    if isdefined(trainable, :fitresult)
        return inverse_transform(trainable.model, trainable.fitresult, X)
    else
        throw(error("$trainable with model $(trainable.model) is not trained and so cannot inverse_transform."))
    end
end

# TODO: predict_proba method for classifier models:


## LEARNING NETWORKS INTERFACE - BASICS

struct Source{D} <: AbstractNode
    data::D      # training data
end

is_stale(s::Source) = false

# make source nodes callable:
(s::Source)() = s.data
(s::Source)(Xnew) = Xnew

struct Node{M<:Union{NodalTrainableModel, Nothing}} <: AbstractNode

    operation::Function   # that can be dispatched on `trainable`(eg, `predict`) or a static operation
    trainable::M          # is `nothing` for static operations
    args::Tuple{Vararg{AbstractNode}}       # nodes where `operation` looks for its arguments
    tape::Vector{NodalTrainableModel}    # for tracking dependencies
    depth::Int64

    function Node{M}(operation, trainable::M, args::AbstractNode...) where {B<:Model, M<:Union{NodalTrainableModel{B},Nothing}}

        # check the number of arguments:
        if trainable == nothing
            length(args) > 0 || throw(error("`args` in `Node(::Function, args...)` must be non-empty. "))
        end

        # get the trainable model's dependencies:
        tape = copy(get_tape(trainable))

        # add the trainable model itself as a dependency:
        if trainable != nothing
            merge!(tape, [trainable, ])
        end

        # append the dependency tapes of all arguments:
        for arg in args
            merge!(tape, get_tape(arg))
        end

        depth = maximum(get_depth(arg) for arg in args) + 1

        return new{M}(operation, trainable, args, tape, depth)

    end
end

# ... where
get_depth(::Source) = 0
get_depth(X::Node) = X.depth

function is_stale(X::Node)
    (X.trainable != nothing && is_stale(X.trainable)) ||
        reduce(|, [is_stale(arg) for arg in X.args])
end

# to complete the definition of `NodalTrainableModel` and `Node`
# constructors:
get_tape(::Any) = NodalTrainableModel[]
get_tape(X::Node) = X.tape
get_tape(trainable::NodalTrainableModel) = trainable.tape

# autodetect type parameter:
Node(operation, trainable::M, args...) where M<:Union{NodalTrainableModel, Nothing} =
    Node{M}(operation, trainable, args...)

# constructor for static operations:
Node(operation::Function, args::AbstractNode...) = Node(operation, nothing, args...)

# note: the following two methods only work as expected if the
# Node `y` has a single source.  TODO: track the source and
# make sure it is unique

# make nodes callable:
(y::Node)() = (y.operation)(y.trainable, [arg() for arg in y.args]...)
(y::Node)(Xnew) = (y.operation)(y.trainable, [arg(Xnew) for arg in y.args]...)

# and for the special case of static operations:
(y::Node{Nothing})() = (y.operation)([arg() for arg in y.args]...)
(y::Node{Nothing})(Xnew) = (y.operation)([arg(Xnew) for arg in y.args]...)

function fit!(y::Node, rows; verbosity=1)
    for trainable in y.tape
        fit!(trainable, rows; verbosity=verbosity-1)
    end
    return y
end
# if no `rows` specified, only retrain stale dependent NodalTrainableModels
# (using whatever rows each was last trained on):
function fit!(y::Node; verbosity=1)
    trainables = filter(is_stale, y.tape)
    for trainable in trainables
        fit!(trainable; verbosity=verbosity-1)
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
                        bold=true, color=color)
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
    
function Base.show(stream::IO, ::MIME"text/plain", trainable::NodalTrainableModel)
    id = objectid(trainable) 
    description = string(typeof(trainable).name.name)
    str = "$description @ $(handle(trainable))"
    printstyled(IOContext(stream, :color=> true), str, bold=true)
    print(stream, " = ")
    print(stream, "trainable($(trainable.model), ")
    n_args = length(trainable.args)
    counter = 1
    for arg in trainable.args
        printstyled(IOContext(stream, :color=>true), handle(arg), bold=true)
        counter >= n_args || print(stream, ", ")
        counter += 1
    end
    print(stream, ")")
end

## SYNTACTIC SUGAR FOR LEARNING NETWORKS

Node(X) = Source(X) # here `X` is data

# aliases
node = Node
trainable = NodalTrainableModel

# TODO: use macro to autogenerate these during model decleration/package glue-code:
# remove need for `Node` syntax in case of operations of main interest:
predict(trainable::NodalTrainableModel, X::AbstractNode) = node(predict, trainable, X)
transform(trainable::NodalTrainableModel, X::AbstractNode) = node(transform, trainable, X)
inverse_transform(trainable::NodalTrainableModel, X::AbstractNode) = node(inverse_transform, trainable, X)

array(X) = convert(Array, X)
array(X::AbstractNode) = node(array, X)

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
