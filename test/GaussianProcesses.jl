module TestGaussianProcesses

using MLJ
using Test

task = load_iris()

X, y = X_and_y(task)
X_array = convert(Array{Float64}, X)

import GaussianProcesses
import CategoricalArrays

baregp = GPClassifier(target_type=String)

# split the rows:
allrows = eachindex(y)
train, test = partition(allrows, 0.7)
@test vcat(train, test) == allrows

fitresult, cache, report = MLJ.fit(baregp, 1, X_array, y)

yhat = predict(baregp, fitresult, X_array)

@show sum(yhat .== y) / length(y)

gp = machine(baregp, X, y)
fit!(gp)
predict(gp, X[1:3, :])

end # module
true
