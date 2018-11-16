abstract type Node <: MLJType end

# a tape is a vector of `TrainableModels` defined below, used to track dependencies
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

mutable struct TrainableModel{B<:Model} <: MLJType

    model::B
    fitresult
    cache
    args::Tuple{Vararg{Node}}
    report
    tape::Vector{TrainableModel}
    frozen::Bool
    
    function TrainableModel{B}(model::B, args::Node...) where B<:Model


        # check number of arguments for model subtypes:
        !(B <: Supervised) || length(args) == 2 ||
            throw(error("Wrong number of arguments. "*
                        "Use TrainableModel(model, X, y) for supervised learner models."))
        !(B <: Unsupervised) || length(args) == 1 ||
            throw(error("Wrong number of arguments. "*
                        "Use TrainableModel(model, X) for an unsupervised learner model."))
        !(B <: Transformer) || length(args) == 1 ||
            throw(error("Wrong number of arguments. Use TrainableModel(model, X) for "*
                        "transformer models."))
        
        trainable = new{B}(model)
        trainable.frozen = false
        trainable.args = args
        trainable.report = Dict{Symbol,Any}()

        # note: `get_tape(arg)` returns arg.tape where this makes
        # sense and an empty tape otherwise.  However, the complete
        # definition of `get_tape` must be postponed until
        # `LearningNode` type is defined.

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
TrainableModel(model::B, args...) where B<:Model = TrainableModel{B}(model, args...)

# to turn of fit-through fitting:
function freeze!(trainable::TrainableModel)
    trainable.frozen = true
end
function thaw!(trainable::TrainableModel)
    trainable.frozen = false
end

# fit method:
function fit!(trainable::TrainableModel, verbosity; kwargs...)

    if trainable.frozen && verbosity > -1
        @warn "$trainable not trained as it is frozen."
        return trainable
    end
        
    verbosity < 1 || @info "Training $trainable whose model is $(trainable.model)."

    args = [arg() for arg in trainable.args]

    if !isdefined(trainable, :fitresult)
        trainable.fitresult, trainable.cache, report =
            fit(trainable.model, verbosity, args...)
    else
        trainable.fitresult, trainable.cache, report =
        update(trainable.model, verbosity, trainable.fitresult, trainable.cache, args...; kwargs...)
    end

    if report != nothing
        merge!(trainable.report, report)
    end

#    verbosity <1 || @info "Done."

    return trainable

end

# for convenience:
fit!(trainable::TrainableModel; kwargs...) = fit!(trainable, 1; kwargs...) 

# predict method for trainable learner models (X data):
function predict(trainable::TrainableModel{L}, X) where L<: Learner 
    if isdefined(trainable, :fitresult)
        return predict(trainable.model, trainable.fitresult, X)
    else
        throw(error("$trainable with model $(trainable.model) is not trained and so cannot predict."))
    end
end

# TODO: predict_proba method for classifier models:

# a transform method for trainable transformer models (X data):
function transform(trainable::TrainableModel{T}, X) where T<:Transformer
    if isdefined(trainable, :fitresult)
        return transform(trainable.model, trainable.fitresult, X)
    else
        throw(error("$trainable with model $(trainable.model) is not trained and so cannot transform."))
    end
end

# an inverse-transform method for trainable transformer models (X data):
function inverse_transform(trainable::TrainableModel{T}, X) where T<:Transformer
    if isdefined(trainable, :fitresult)
        return inverse_transform(trainable.model, trainable.fitresult, X)
    else
        throw(error("$trainable with model $(trainable.model) is not trained and so cannot inverse_transform."))
    end
end


## LEARNING NETWORKS INTERFACE - BASICS

# TODO: do these really need to be mutable?
mutable struct SourceNode{D} <: Node
    data::D      # training data
end

# make source nodes callable:
(s::SourceNode)() = s.data
(s::SourceNode)(Xnew) = Xnew

struct LearningNode{M<:Union{TrainableModel, Nothing}} <: Node

    operation::Function   # that can be dispatched on `trainable`(eg, `predict`) or a static operation
    trainable::M          # is `nothing` for static operations
    args::Tuple{Vararg{Node}}       # nodes where `operation` looks for its arguments
    tape::Vector{TrainableModel}    # for tracking dependencies
    depth::Int64       

    function LearningNode{M}(operation, trainable::M, args::Node...) where {B<:Model, M<:Union{TrainableModel{B},Nothing}}

        # check the number of arguments:
        if trainable == nothing
            length(args) > 0 || throw(error("`args` in `LearningNode(::Function, args...)` must be non-empty. "))
        elseif B<:Union{Learner,Transformer}
            length(args) == 1 || throw(error("Wrong number of arguments. "*
                                             "Use `LearningNode(operation, trainable_model, X)` "*
                                             "for learner or transformer models."))
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
get_depth(::SourceNode) = 0
get_depth(X::LearningNode) = X.depth

# to complete the definition of `TrainableModel` and `LearningNode`
# constructors:
get_tape(::Any) = TrainableModel[]
get_tape(X::LearningNode) = X.tape
get_tape(trainable::TrainableModel) = trainable.tape

# autodetect type parameter:
LearningNode(operation, trainable::M, args...) where M<:Union{TrainableModel, Nothing} =
    LearningNode{M}(operation, trainable, args...)

# constructor for static operations:
LearningNode(operation::Function, args::Node...) = LearningNode(operation, nothing, args...)

# note: the following two methods only work as expected if the
# LearningNode `y` has a single source.  TODO: track the source and
# make sure it is unique

# make learning nodes callable:
(y::LearningNode)() = (y.operation)(y.trainable, [arg() for arg in y.args]...)
(y::LearningNode)(Xnew) = (y.operation)(y.trainable, [arg(Xnew) for arg in y.args]...)

# and for the special case of static operations:
(y::LearningNode{Nothing})() = (y.operation)([arg() for arg in y.args]...)
(y::LearningNode{Nothing})(Xnew) = (y.operation)([arg(Xnew) for arg in y.args]...)

# the "fit through" method:
function fit!(y::LearningNode, verbosity; kwargs...)
    for trainable in y.tape[1:end-1]
        fit!(trainable, verbosity)
    end
    fit!(y.tape[end], verbosity; kwargs...)
    return y
end

# for convenience:
fit!(y::LearningNode; kwargs...) = fit!(y, 1; kwargs...)

# allow arguments of `LearningNodes` and `TrainableModel`s to appear
# at REPL:
istoobig(d::Tuple{Node}) = length(d) > 10

# overload show method
function spaces(n)
    s = ""
    for i in 1:n
        s = string(s, " ")
    end
    return s
end
function Base.show(stream::IO, ::MIME"text/plain", X::LearningNode)
#   gap = spaces(20 - TREE_INDENT*get_depth(X) + TREE_INDENT)
    gap = ""
    if X isa SourceNode
        @show "here"
        print(stream, gap)
    else
        detail = (X.trainable == nothing ? "(" : "($(handle(X.trainable)), ")
        operation_name = typeof(X.operation).name.mt.name
#       println(stream, gap, operation_name, detail) # or uncomment next two lines
        print(stream, operation_name, detail)
        n_args = length(X.args)
        counter = 1
        for arg in X.args
            if arg isa LearningNode
                show(stream, MIME("text/plain"), arg)
            else # arg is a SourceNode, and so:
#               print(stream, gap, spaces(TREE_INDENT), handle(arg))
                printstyled(IOContext(stream, :color=> true), handle(arg), color=:blue)
            end
            counter >= n_args || print(stream, ", ")
            counter += 1
        end
    print(stream, ")")
    end
end


## SYNTACTIC SUGAR FOR LEARNING NETWORKS

LearningNode(X) = SourceNode(X) # here `X` is data

# aliases
node = LearningNode
trainable = TrainableModel

# remove need for `LearningNode` syntax in case of operations of main interest:
predict(trainable::TrainableModel{L}, X::Node) where L<:Learner =
    node(predict, trainable, X)
transform(trainable::TrainableModel{T}, X::Node) where T<:Transformer =
    node(transform, trainable, X)
inverse_transform(trainable::TrainableModel{T}, X::Node) where T<:Transformer =
    node(inverse_transform, trainable, X)

array(X) = convert(Array, X)
array(X::Node) = node(array, X)

Base.log(v::Vector{<:Number}) = log.(v)
Base.exp(v::Vector{<:Number}) = exp.(v)
Base.log(X::Node) = node(log, X)
Base.exp(X::Node) = node(exp, X)


rms(y::Node, yhat::Node) = node(rms, y, yhat)
rms(y, yhat::Node) = node(rms, y, yhat)
rms(y::Node, yhat) = node(rms, y, yhat)

import Base.+
+(y1::Node, y2::Node) = node(+, y1, y2)
+(y1, y2::Node) = node(+, y1, y2)
+(y1::Node, y2) = node(+, y1, y2)
