module TestScitypes

using MLJ
import MLJBase

scitype(ConstantClassifier())
scitype(FeatureSelector())

for handle in localmodels()
    name = Symbol(handle.name)
    eval(quote
         scitype(($name)())
         end)
end

end
true
