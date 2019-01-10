module TestMetrics

# using Revise
using Test
using MLJ
import Distributions
using CategoricalArrays

## REGRESSOR METRICS

y = [1, 2, 3, 4]
yhat = y .+ 1
@test isapprox(rms(y, yhat), 1.0)
@test isapprox(rmsl(y, yhat),
               sqrt((log(1/2)^2 + log(2/3)^2 + log(3/4)^2 + log(4/5)^2)/4))
@test isapprox(rmslp1(y, yhat),
               sqrt((log(2/3)^2 + log(3/4)^2 + log(4/5)^2 + log(5/6)^2)/4))
@test isapprox(rmsp(y, yhat), sqrt((1 + 1/4 + 1/9 + 1/16)/4))

# probababilistic versions:
N = Distributions.Normal
zhat = N.(yhat)
@test isapprox(rms(y, zhat), 1.0)
@test isapprox(rmsl(y, zhat),
               sqrt((log(1/2)^2 + log(2/3)^2 + log(3/4)^2 + log(4/5)^2)/4))
@test isapprox(rmslp1(y, zhat),
               sqrt((log(2/3)^2 + log(3/4)^2 + log(4/5)^2 + log(5/6)^2)/4))
@test isapprox(rmsp(y, zhat), sqrt((1 + 1/4 + 1/9 + 1/16)/4))


## CLASSIFIER METRICS

y    = categorical(collect("asdfasdfaaassdd"))
yhat = categorical(collect("asdfaadfaasssdf"))
@test misclassification_rate(y, yhat) ≈ 0.2

y = categorical(collect("abb"))
L = ['a', 'b']
d1 = UnivariateNominal(L, [0.1, 0.9])
d2 = UnivariateNominal(L, [0.4, 0.6])
d3 = UnivariateNominal(L, [0.2, 0.8])
yhat = [d1, d2, d3]
@test cross_entropy(y, yhat) ≈ -(log(0.1) + log(0.6) + log(0.8))/3



# for when ROC is added as working dependency:
# y = ["n", "p", "n", "p", "n", "p"]
# yhat = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]
# @test auc("p")(y, yhat) ≈ 2/3
end
true
