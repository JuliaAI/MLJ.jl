module TestMetrics

# using Revise
using Test
using MLJ
import Distributions
using CategoricalArrays

## REGRESSOR METRICS

y = [1, 2, 3, 4]
yhat = y .+ 1
@test isapprox(rms(yhat, y), 1.0)
@test isapprox(rmsl(yhat, y),
               sqrt((log(1/2)^2 + log(2/3)^2 + log(3/4)^2 + log(4/5)^2)/4))
@test isapprox(rmslp1(yhat, y),
               sqrt((log(2/3)^2 + log(3/4)^2 + log(4/5)^2 + log(5/6)^2)/4))
@test isapprox(rmsp(yhat, y), sqrt((1 + 1/4 + 1/9 + 1/16)/4))

# probababilistic versions:
N = Distributions.Normal
zhat = N.(yhat)
@test isapprox(rms(zhat, y), 1.0)
@test isapprox(rmsl(zhat, y),
               sqrt((log(1/2)^2 + log(2/3)^2 + log(3/4)^2 + log(4/5)^2)/4))
@test isapprox(rmslp1(zhat, y),
               sqrt((log(2/3)^2 + log(3/4)^2 + log(4/5)^2 + log(5/6)^2)/4))
@test isapprox(rmsp(zhat, y), sqrt((1 + 1/4 + 1/9 + 1/16)/4))


## CLASSIFIER METRICS

y    = categorical(collect("asdfasdfaaassdd"))
yhat = categorical(collect("asdfaadfaasssdf"))
@test misclassification_rate(yhat, y) ≈ 0.2

# probabilistic version:
d= MLJ.fit(UnivariateFinite, y[end-2:end]) # mode of `d`
yhat = fill(d, length(y))
@test misclassification_rate(yhat, y) ≈ 11/15

y = categorical(collect("abb"))
L = ['a', 'b']
d1 = UnivariateFinite(L, [0.1, 0.9])
d2 = UnivariateFinite(L, [0.4, 0.6])
d3 = UnivariateFinite(L, [0.2, 0.8])
yhat = [d1, d2, d3]
@test cross_entropy(yhat, y) ≈ -(log(0.1) + log(0.6) + log(0.8))/3



# for when ROC is added as working dependency:
# y = ["n", "p", "n", "p", "n", "p"]
# yhat = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]
# @test auc("p")(yhat, y) ≈ 2/3
end
true
