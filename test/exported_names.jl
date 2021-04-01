# MLJIteration

using MLJ

for control in MLJ.MLJIteration.CONTROLS
    eval(:($control))
end

IterationControl.skip(Step(2))
IterationControl.debug(Step(2))

IteratedModel


# MLJSerialization

Save()
@test MLJ.save isa Function


# MLJOpenML

@test OpenML.load isa Function

true
