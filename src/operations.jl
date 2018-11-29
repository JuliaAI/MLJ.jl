# methods to be dispatched on suitable `Model`s and a fit-result (*operations*):
function predict end
function predict_proba end
function transform end 
function inverse_transform end
function se end
function evaluate end

# We wish to extend these to identically named operations dispatched
# on `TrainableModels` and `NodalTrainableModel`. For example, we have
#
# `predict(model::M, fitresult, X) where M<:Supervised`
#
# but want also want
#
# `predict(trainable_model::TrainableModel, X)` where `X` is data
#
# and "networks.jl" requires us to define
#
# `predict(trainable_model::NodalTrainableModel, X)` where `X` is data
#
# and we would like the syntactic sugar
#
# `predict(trainable_model::NodalTrainableModel, X::Node)=node(predict, trainable_model, X)`
#
# (If an operation has zero arguments, we cannot achieve the last
# desire because of ambiguity with the preceding one.) The following
# macro is for this purpose.

macro extend_to_trainable_models(operation)
    quote
        
        function $(esc(operation))(trainable_model::AbstractTrainableModel, args...) 
            if isdefined(trainable_model, :fitresult)
                tst = trainable_model isa Supervised
                return $(esc(operation))(trainable_model.model,
                                         trainable_model.fitresult,
                                         coerce(trainable_model.model, args...)...)
            else
                throw(error("$trainable_model with model $(trainable_model.model) is not trained and so cannot predict."))
            end
        end

        $(esc(operation))(trainable_model::NodalTrainableModel, args::AbstractNode...) =
            node($(esc(operation)), trainable_model, args...)
        
    end
end

@extend_to_trainable_models predict
@extend_to_trainable_models transform
@extend_to_trainable_models inverse_transform
@extend_to_trainable_models predict_proba
@extend_to_trainable_models se
# @extend_to_trainable_models evaluate



