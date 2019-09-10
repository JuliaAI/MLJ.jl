using MLJ
using Test

X = (a = rand(5), b = categorical(1:5))
y = rand(5)
w = rand(5)

@test matching(X) == Main.ModelChecker{false,false,scitype(X),missing}()
@test matching(X, y) == Main.ModelChecker{true,false,scitype(X),scitype(y)}()
@test matching(X, y, w) == Main.ModelChecker{true,true,scitype(X),scitype(y)}()

@test !matching("RidgeRegressor", pkg="MultivariateStats", X)
@test matching("FeatureSelector", X)

m1 = models(matching(X))
@test issubset([info("FeatureSelector"),
               info("OneHotEncoder"),
                info("Standardizer")], m1)

@test !("PCA" in m1)
@test !(info("PCA") in m1)

m2 = models(matching(X, y))
matching(X, y)("ConstantRegressor")
matching(X,y)
