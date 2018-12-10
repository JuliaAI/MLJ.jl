# We wish to extend operations to identically named methods dispatched
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
# and we would like the syntactic sugar (for `X` a node):
#
# `predict(trainable_model::NodalTrainableModel, X::Node)=node(predict, trainable_model, X)`
#
# (If an operation has zero arguments, we cannot achieve the last
# desire because of ambiguity with the preceding one.)

# The following macro is for this purpose.

macro extend_to_trainable_models(operation)
    quote

        # most general (no coersion):
        function $(esc(operation))(trainable_model::AbstractTrainableModel, args...) 
            if isdefined(trainable_model, :fitresult)
                tst = trainable_model isa Supervised
                return $(esc(operation))(trainable_model.model,
                                         trainable_model.fitresult,
                                         args...)
            else
#                throw(error("$trainable_model with model $(trainable_model.model) is not trained and so cannot predict."))
                throw(error("$trainable_model is not trained and so cannot predict."))
            end
        end

        # for supervised models (data must be coerced):
        function $(esc(operation))(trainable_model::AbstractTrainableModel{M}, Xtable) where M<:Supervised
            if isdefined(trainable_model, :fitresult)
                return $(esc(operation))(trainable_model.model,
                                         trainable_model.fitresult,
                                         coerce(trainable_model.model, Xtable))
            else
#                throw(error("$trainable_model with model $(trainable_model.model) is not trained and so cannot predict."))
                throw(error("$trainable_model is not trained and so cannot predict."))
            end
        end
    end
end

macro sugar(operation)
    quote
        $(esc(operation))(trainable_model::NodalTrainableModel, args::AbstractNode...) =
            node($(esc(operation)), trainable_model, args...)
    end
end


@extend_to_trainable_models predict
@extend_to_trainable_models transform
@extend_to_trainable_models inverse_transform
@extend_to_trainable_models predict_proba
@extend_to_trainable_models se
@extend_to_trainable_models evaluate
@extend_to_trainable_models best

@sugar predict
@sugar transform
@sugar inverse_transform
@sugar predict_proba
@sugar se



