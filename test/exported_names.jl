# MLJIteration

using MLJ

for control in MLJ.MLJIteration.CONTROLS
    eval(:($control))
end

IterationControl.skip(Step(2))
IterationControl.with_state_do(Step(2))

IteratedModel
MLJIteration

# MLJBalancing

bmodel = @test_logs(
    (:warn, r"^No balancer"),
    BalancedModel(model=ConstantClassifier()),
)

@test bmodel isa Probabilistic


# MLJSerialization

Save()
@test MLJ.save isa Function


# MLJOpenML

@test OpenML.load isa Function


# MLJFlow

MLJFlow.Logger

# StatisticalMeasures

rms
l2
log_score

true
