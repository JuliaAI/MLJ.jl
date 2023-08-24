# MLJIteration

using MLJ

for control in MLJ.MLJIteration.CONTROLS
    eval(:($control))
end

IterationControl.skip(Step(2))
IterationControl.with_state_do(Step(2))

IteratedModel
MLJIteration

# MLJSerialization

Save()
@test MLJ.save isa Function


# MLJOpenML

@test OpenML.load isa Function


# MLJFlow

MLFlowLogger

true
