module TestTransformer

# using Revise
using MLJ
using Test
using Statistics
using DataFrames

# selecting features
Xtable, y = datanow()
X = DataFrame(Xtable) # will become redunant in DataFrames 0.17.0
namesX = names(X)
selector = FeatureSelector()
fitresult, cache, report = MLJ.fit(selector, 1, X)
@test fitresult == namesX
transform(selector, fitresult, X[1:2,:])
selector = FeatureSelector([:Zn, :Crim])
fitresult, cache, report = MLJ.fit(selector, 1, X)
@test names(transform(selector, fitresult, X[1:2,:])) == [:Zn, :Crim]

# relabelling with integer transformer:
y = rand(Char, 50)
allrows = eachindex(y)
test = 3:37
to_int_hypers = ToIntTransformer()
fitresult, cache, report = MLJ.fit(to_int_hypers, 1, y)
# to_int = Trainable(to_int_hypers, y)
# fit!(to_int, allrows)
z = transform(to_int_hypers, fitresult, y[test])
@test y[test] == inverse_transform(to_int_hypers, fitresult, z)
to_int_hypers.map_unseen_to_minus_one = true
fitresult, cache, report = MLJ.fit(to_int_hypers, 1, [1, 2, 3, 4, 3])
@test report[:values] == [1, 2, 3, 4]
#to_int = Trainable(to_int_hypers, [1,2,3,4])
#fitresult, cache, report = fit!(to_int, [1,2,3,4])
@test transform(to_int_hypers, fitresult, 5) == -1
@test transform(to_int_hypers, fitresult, [5,1])[1] == -1 

# `UnivariateStandardizer`:
stand = UnivariateStandardizer()
#fit!(stand, 1:3)
fitresult, cache, report = MLJ.fit(stand, 1, [0, 2, 4])
@test round.(Int, transform(stand, fitresult, [0,4,8])) == [-1.0,1.0,3.0]
@test round.(Int, inverse_transform(stand, fitresult, [-1, 1, 3])) == [0, 4, 8] 

# `Standardizer`:
X, y = X_and_y(load_ames())
X = DataFrame(X) # will be redundant in DataFrames 0.17.0
X = X[1:4,1:4]
train, test = partition(eachindex(y), 0.9);

# introduce a field of type `Char`:
X[:OverallQual] = map(Char, X[:OverallQual]);

# introduce field of Int type:
X[:x1stFlrSF] = [round(Int, x) for x in X[:x1stFlrSF]]

stand = Standardizer()
fitresult, cache, report = MLJ.fit(stand, 1, X)
Xnew = transform(stand, fitresult, X)
@test std(Xnew[2]) ≈ 1.0
@test Xnew[1] == X[1]
@test Xnew[3] == X[3]
@test Xnew[4] == X[4]

stand.features = names(X)
fitresult, cache, report = MLJ.fit(stand, 1, X)
Xnew = transform(stand, fitresult, X)

fitresult, cache, report = MLJ.fit(stand, 1, X)
@test issubset(Set(keys(fitresult)), Set(names(X)[[2,4]]))
transform(stand, fitresult, X)
@test Xnew[1] == X[1]
@test Xnew[3] == X[3]
@test std(Xnew[2]) ≈ 1.0
@test std(Xnew[4]) ≈ 1.0

# `UnivariateBoxCoxTransformer`

# create skewed non-negative vector with a zero value:
v = abs.(randn(1000))
v = v .- minimum(v)
MLJ.Transformers.normality(v)

t = UnivariateBoxCoxTransformer(shift=true)
fitresult, cache, report = MLJ.fit(t, 2, v)
@test sum(abs.(v - MLJ.inverse_transform(t, fitresult, MLJ.transform(t, fitresult, v)))) <= 5000*eps()

end
true
