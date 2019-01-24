module TestMultivariateStats

using MLJ
using Test
using LinearAlgebra: svd

task = load_crabs()

X, y = X_and_y(task)

import MultivariateStats

barepca = PCA(pratio=0.9999)

fitresult, cache, report = MLJ.fit(barepca, 1, X)

Xtr = MLJ.transform(barepca, fitresult, X)

X_array = convert(Matrix{Float64}, X)

# home made PCA (the sign flip is irrelevant)
Xac = X_array .- mean(X_array, dims=1)
U, S, _ = svd(Xac)
Xtr_ref = abs.(U .* S')
@test abs.(Xtr) ≈ Xtr_ref

# machinery
pca = machine(barepca, X)
fit!(pca)

Xtr2 = transform(barepca, fitresult, X)
@test abs.(Xtr2) ≈ Xtr_ref

end # module
true
