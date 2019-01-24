module TestClustering

using MLJ
using Test
using Random:seed!
using LinearAlgebra:norm

seed!(132442)

task = load_crabs()

X, y = X_and_y(task)


import Clustering

barekm = KMeans()

fitresult, cache, report = MLJ.fit(barekm, 1, X)

r = MLJ.transform(barekm, fitresult, X)

X_array = convert(Matrix{Float64}, X)

# distance from first point to second center
@test r[1, 2] == norm(view(X_array, 1, :) .- view(fitresult.centers, :, 2))
@test r[10, 3] == norm(view(X_array, 10, :) .- view(fitresult.centers, :, 3))

p = MLJ.predict(barekm, fitresult, X)

@test argmin(r[1, :]) == p[1]
@test argmin(r[10, :]) == p[10]

km = machine(barekm, X)

fit!(km)

end # module
true
