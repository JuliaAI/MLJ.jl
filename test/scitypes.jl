module TestScitypes

using MLJ
import MLJBase
using Test

S = scitype(ConstantClassifier())
@test S().prediction_type == :probabilistic
U = scitype(FeatureSelector())
@test U().input_scitype == MLJ.Table(Scientific)

# XXX fix this in the next release cycle (Nov 5, 2019)
# M = scitype(rms)
# @test_broken M().prediction_type == :deterministic

for handle in localmodels()
    name = Symbol(handle.name)
    eval(quote
         scitype(($name)())
         end)
end

end
true
