module TestGLM

# using Revise
using MLJ
using Test

task = load_boston()
X, y = X_and_y(task)

import GLM

train, test = partition(eachindex(y), 0.7)

atom_ols = OLSRegressor()

ols = machine(atom_ols, MLJ.selectrows(X, train), y[train])
fit!(ols)

p = predict(ols, MLJ.selectrows(X, test))

# hand made regression to compare

Xa = MLJ.matrix(X) # convert(Matrix{Float64}, X)
Xa1 = hcat(Xa, ones(size(Xa, 1)))
coefs = Xa1[train, :] \ y[train]

p2 = Xa1[test, :] * coefs

@test p ≈ p2

info(atom_ols)

end # module
true
