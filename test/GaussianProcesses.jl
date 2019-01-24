module TestGaussianProcesses

using MLJ
using Test
using Random:seed!, shuffle!

seed!(113355)

task = load_crabs()

X, y = X_and_y(task)
X_array = convert(Matrix{Float64}, X)

import GaussianProcesses
import CategoricalArrays

baregp = GPClassifier(target_type=String)

# split the rows:
allrows = eachindex(y)
train, test = partition(allrows, 0.7, shuffle=true)
@test sort(vcat(train, test)) == allrows

fitresult, cache, report = MLJ.fit(baregp, 1, X_array[train,:], y[train])

yhat = predict(baregp, fitresult, X_array[test, :])

@test sum(yhat .== y[test]) / length(y[test]) >= 0.7 # around 0.7

gp = machine(baregp, X, y)
fit!(gp)
yhat2 = predict(gp, X[test,:])

@test sum(yhat2 .== y[test]) / length(y[test]) >= 0.7

end # module
true
