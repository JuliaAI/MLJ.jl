module TestTransformer

# using Revise
import MLJ
using MLJ.Transformers
using MLJBase
using Test
using Statistics
using CategoricalArrays
using Tables

# selecting features
N =100
X = (Zn=rand(N),
     Crim=rand(N),
     x3=categorical(rand("yn",N)),
     x4=categorical(rand("yn",N)))

namesX = schema(X).names |> collect
selector = FeatureSelector()
info(selector)
fitresult, cache, report = MLJBase.fit(selector, 1, X)
@test fitresult == namesX
transform(selector, fitresult, selectrows(X, 1:2))
selector = FeatureSelector([:Zn, :Crim])
fitresult, cache, report = MLJBase.fit(selector, 1, X)
@test transform(selector, fitresult, selectrows(X, 1:2)) ==
    selectcols(selectrows(X, 1:2), [:Zn, :Crim])

# `UnivariateStandardizer`:
stand = UnivariateStandardizer()
info(stand)
#fit!(stand, 1:3)
fitresult, cache, report = MLJBase.fit(stand, 1, [0, 2, 4])
@test round.(Int, transform(stand, fitresult, [0,4,8])) == [-1.0,1.0,3.0]
@test round.(Int, inverse_transform(stand, fitresult, [-1, 1, 3])) == [0, 4, 8] 

# `Standardizer`:
N = 5
X = (OverallQual=rand(UInt8, N),
     GrLivArea=rand(N),
     Neighborhood=categorical(rand("abc", N)),
     x1stFlrSF=rand(N),
     TotalBsmtSF=rand(N))

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

X = (name=categorical(["Ben", "John", "Mary", "John"], ordered=true),
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
@test schema(Xt).names == (:name__Ben, :name__John, :name__Mary,
                           :height, :favourite_number__5,
                           :favourite_number__7, :favourite_number__10,
                           :age) 

# test that *entire* pool of categoricals is used in fit, including
# unseen levels:
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
X = (name=categorical(["Ben", "John", "Mary", "John"], ordered=true),
     height=[1.85, 1.67, 1.5, 1.67],
     favourite_number=categorical([7, 5, 10, 5]),
     age=[23, 23, 14, 23],
     gender=categorical(['M', 'M', 'F', 'M']))
@test_throws Exception transform(t, fitresult, X)

#

end
true
