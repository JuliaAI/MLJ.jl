module TestTransformer

# using Revise
using MLJ, MLJBase
using Test
using Statistics
using DataFrames
using CategoricalArrays
using Tables

# selecting features
Xtable, y = datanow()
X = DataFrame(Xtable) # will become redunant in DataFrames 0.17.0
namesX = names(X)
selector = FeatureSelector()
info(selector)
fitresult, cache, report = MLJBase.fit(selector, 1, X)
@test fitresult == namesX
transform(selector, fitresult, X[1:2,:])
selector = FeatureSelector([:Zn, :Crim])
fitresult, cache, report = MLJBase.fit(selector, 1, X)
@test transform(selector, fitresult, X[1:2,:]) ==
    MLJBase.selectcols(MLJBase.selectrows(X, 1:2), [:Zn, :Crim])

# # relabelling with integer transformer:
# y = rand(Char, 50)
# allrows = eachindex(y)
# test = 3:37
# to_int_hypers = ToIntTransformer()
# info(to_int_hypers)
# fitresult, cache, report = MLJBase.fit(to_int_hypers, 1, y)
# # to_int = Trainable(to_int_hypers, y)
# # fit!(to_int, allrows)
# z = transform(to_int_hypers, fitresult, y[test])
# @test y[test] == inverse_transform(to_int_hypers, fitresult, z)
# to_int_hypers.map_unseen_to_minus_one = true
# fitresult, cache, report = MLJBase.fit(to_int_hypers, 1, [1, 2, 3, 4, 3])
# @test report.values == [1, 2, 3, 4]
# #to_int = Trainable(to_int_hypers, [1,2,3,4])
# #fitresult, cache, report = fit!(to_int, [1,2,3,4])
# @test transform(to_int_hypers, fitresult, 5) == -1
# @test transform(to_int_hypers, fitresult, [5,1])[1] == -1 

# `UnivariateStandardizer`:
stand = UnivariateStandardizer()
info(stand)
#fit!(stand, 1:3)
fitresult, cache, report = MLJBase.fit(stand, 1, [0, 2, 4])
@test round.(Int, transform(stand, fitresult, [0,4,8])) == [-1.0,1.0,3.0]
@test round.(Int, inverse_transform(stand, fitresult, [-1, 1, 3])) == [0, 4, 8] 

# `Standardizer`:
X, y = X_and_y(load_reduced_ames())
X = selectrows(X, 1:5)
X = selectcols(X, 1:5)
train, test = partition(eachindex(y), 0.9);

# introduce a field of type `Char`:
x1 = categorical(map(Char, (X.OverallQual |> collect)))

# introduce field of Int type:
x4 = [round(Int, x) for x in X.x1stFlrSF]

X = (x1=x1, x2=X[2], x3=X[3], x4=x4, x5=X[5])

stand = Standardizer()
info(stand)
fitresult, cache, report = MLJBase.fit(stand, 1, X)
Xnew = transform(stand, fitresult, X)
@test Xnew[1] == X[1]
@test std(Xnew[2]) ≈ 1.0
@test Xnew[3] == X[3]
@test Xnew[4] == X[4]
@test std(Xnew[5]) ≈ 1.0

stand.features = [:x1, :x5]
fitresult, cache, report = MLJBase.fit(stand, 1, X)
Xnew = transform(stand, fitresult, X)

fitresult, cache, report = MLJBase.fit(stand, 1, X)
@test issubset(Set(keys(fitresult)), Set(MLJBase.schema(X).names[[5,]]))
transform(stand, fitresult, X)
@test Xnew[1] == X[1]
@test Xnew[2] == X[2]
@test Xnew[3] == X[3]
@test Xnew[4] == X[4]
@test std(Xnew[5]) ≈ 1.0

# `UnivariateBoxCoxTransformer`

# create skewed non-negative vector with a zero value:
v = abs.(randn(1000))
v = v .- minimum(v)
MLJ.Transformers.normality(v)

t = UnivariateBoxCoxTransformer(shift=true)
info(t)
fitresult, cache, report = MLJBase.fit(t, 2, v)
@test sum(abs.(v - MLJBase.inverse_transform(t, fitresult, MLJBase.transform(t, fitresult, v)))) <= 5000*eps()


# `OneHotEncoder`

X = DataFrame(name=identity.(categorical(["Ben", "John", "Mary", "John"], ordered=true)),
              height=[1.85, 1.67, 1.5, 1.67],
              favourite_number=categorical([7, 5, 10, 5]),
              age=[23, 23, 14, 23])

t = OneHotEncoder()
info(t)
fitresult, cache, _ =
    @test_logs((:info, r"Spawning 3"),
               (:info, r"Spawning 3"),
               MLJBase.fit(t, 1, X))
Xt = transform(t, fitresult, X)
@test Xt.name__John == float.([false, true, false, true])
@test Xt.height == X.height
@test Xt.favourite_number__10 == float.([false, false, true, false])
@test Xt.age == X.age
@test Tables.schema(Xt).names == (:name__Ben, :name__John, :name__Mary,
                                  :height, :favourite_number__5,
                                  :favourite_number__7, :favourite_number__10, :age) 

# test that *entire* pool of categoricals is used in fit, including unseen levels:
fitresult_small, cache, _ =
    @test_logs((:info, r"Spawning 3"),
               (:info, r"Spawning 3"),
               MLJBase.fit(t, 1, MLJBase.selectrows(X,1:2)))
Xtsmall = transform(t, fitresult_small, X)
@test Xt == Xtsmall

# test that transform can be applied to subset of the data:
@test transform(t, fitresult, MLJBase.selectcols(X, [:name, :age])) ==
    MLJBase.selectcols(transform(t, fitresult, X),
                       [:name__Ben, :name__John, :name__Mary, :age])

# test exclusion of ordered factors:
t = OneHotEncoder(ordered_factor=false)
fitresult, cache, _ = MLJBase.fit(t, 1, X)
Xt = transform(t, fitresult, X)
@test :name in MLJ.schema(Xt).names
@test :favourite_number__5 in MLJ.schema(Xt).names

# test that one may not add new columns:
X.gender = categorical(['M', 'M', 'F', 'M'])
@test_throws Exception transform(t, fitresult, X)

#

end
true
