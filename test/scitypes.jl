module TestScitypes

# using Revise
using MLJ
import MLJBase
using Test

S = scitype(ConstantClassifier())
@test S().prediction_type == :probabilistic
U = scitype(FeatureSelector())
@test U().input_scitype == MLJ.Table(Scientific)
M = scitype(rms)
@test M().prediction_type == :deterministic

for handle in localmodels()
    name = Symbol(handle.name)
    eval(quote
         scitype(($name)())
         end)
end

end
true
