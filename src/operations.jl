# We wish to extend operations to identically named methods dispatched
# on `Machines` and `NodalMachine`. For example, we have
#
# `predict(model::M, fitresult, X) where M<:Supervised`
#
# but want also want
#
# `predict(machine::Machine, X)` where `X` is data
#
# and "networks.jl" requires us to define
#
# `predict(machine::NodalMachine, X)` where `X` is data
#
# and we would like the syntactic sugar (for `X` a node):
#
# `predict(machine::NodalMachine, X::Node)=node(predict, machine, X)`
#
# (If an operation has zero arguments, we cannot achieve the last
# desire because of ambiguity with the preceding one.)

# The following macro is for this purpose.

macro extend_to_machines(operation)
    quote

        # most general (no coersion):
        function $(esc(operation))(machine::AbstractMachine, args...) 
            if isdefined(machine, :fitresult)
                tst = machine isa Supervised
                return $(esc(operation))(machine.model,
                                         machine.fitresult,
                                         args...)
            else
#                throw(error("$machine with model $(machine.model) is not trained and so cannot predict."))
                throw(error("$machine is not trained and so cannot predict."))
            end
        end

        # for supervised models (data must be coerced):
        function $(esc(operation))(machine::AbstractMachine{M}, Xtable) where M<:Supervised
            if isdefined(machine, :fitresult)
                return $(esc(operation))(machine.model,
                                         machine.fitresult,
                                         coerce(machine.model, Xtable))
            else
#                throw(error("$machine with model $(machine.model) is not trained and so cannot predict."))
                throw(error("$machine is not trained and so cannot predict."))
            end
        end
    end
end

macro sugar(operation)
    quote
        $(esc(operation))(machine::NodalMachine, args::AbstractNode...) =
            node($(esc(operation)), machine, args...)
    end
end

@extend_to_machines predict
@extend_to_machines predict_mode
@extend_to_machines predict_mean
@extend_to_machines predict_median
@extend_to_machines transform
@extend_to_machines inverse_transform
@extend_to_machines se
@extend_to_machines evaluate
@extend_to_machines best

@sugar predict
@sugar predict_mode
@sugar predict_mean
@sugar predict_median
@sugar transform
@sugar inverse_transform
@sugar se



