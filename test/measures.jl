module TestMeasures

# using Revise
using Test
using MLJ
import Distributions
using CategoricalArrays
import Random.seed!
seed!(1234)

@testset "built-in regressor measures" begin

    y    = [1, 2, 3, 4]
    yhat = [4, 3, 2, 1]
    w = [1, 2, 4, 3]
    @test isapprox(mav(yhat, y), 2)
    @test isapprox(mav(yhat, y, w), 9/5)
    @test isapprox(rms(yhat, y), sqrt(5))
    @test isapprox(rms(yhat, y, w), sqrt(21/5))
    @test isapprox(mean(l1(yhat, y)), 2)
    @test isapprox(mean(l1(yhat, y, w)), 9/5)
    @test isapprox(mean(l2(yhat, y)), 5)
    @test isapprox(mean(l2(yhat, y, w)), 21/5)
                   
    yhat = y .+ 1
    @test isapprox(rmsl(yhat, y),
                   sqrt((log(1/2)^2 + log(2/3)^2 + log(3/4)^2 + log(4/5)^2)/4))
    @test isapprox(rmslp1(yhat, y),
                   sqrt((log(2/3)^2 + log(3/4)^2 + log(4/5)^2 + log(5/6)^2)/4))
    @test isapprox(rmsp(yhat, y), sqrt((1 + 1/4 + 1/9 + 1/16)/4))

end

@testset "built-in classifier measures" begin

    y    = categorical(collect("asdfasdfaaassdd"))
    yhat = categorical(collect("asdfaadfaasssdf"))
    w = 1:15
    @test misclassification_rate(yhat, y) ≈ 0.2
    @test misclassification_rate(yhat, y, w) ≈ 4/15
    y = categorical(collect("abb"))
    L = [y[1], y[2]]
    d1 = UnivariateFinite(L, [0.1, 0.9])
    d2 = UnivariateFinite(L, [0.4, 0.6])
    d3 = UnivariateFinite(L, [0.2, 0.8])
    yhat = [d1, d2, d3]
    @test mean(cross_entropy(yhat, y)) ≈ -(log(0.1) + log(0.6) + log(0.8))/3

end

@testset "MLJ.value" begin

    yhat = rand(5)
    X = (weight=rand(5), x1 = rand(5))
    y = rand(5)
    w = rand(5)

    @test MLJ.value(mav, yhat, nothing, y, nothing) ≈ mav(yhat, y) 
    @test MLJ.value(mav, yhat, nothing, y, w) ≈ mav(yhat, y, w) 

    spooky(yhat, y) = abs.(yhat - y) |> mean
    @test MLJ.value(spooky, yhat, nothing, y, nothing) ≈ mav(yhat, y)
    
    cool(yhat, y, w) = abs.(yhat - y) .* w ./ (sum(w)/length(y)) |> mean
    MLJ.supports_weights(::typeof(cool)) = true
    @test MLJ.value(cool, yhat, nothing, y, w) ≈ mav(yhat, y, w)
    
    funky(yhat, X, y) = X.weight .* abs.(yhat - y) ./ (sum(X.weight)/length(y)) |> mean
    MLJ.is_feature_dependent(::typeof(funky)) = true
    @test MLJ.value(funky, yhat, X, y, nothing) ≈ mav(yhat, y, X.weight)

    weird(yhat, X, y, w) = w .* X.weight .* abs.(yhat - y) ./ sum(w .* X.weight) |> sum
    MLJ.is_feature_dependent(::typeof(weird)) = true
    MLJ.supports_weights(::typeof(weird)) = true
    @test MLJ.value(weird, yhat, X, y, w) ≈ mav(yhat, y, X.weight .* w)

end



# for when ROC is added as working dependency:
# y = ["n", "p", "n", "p", "n", "p"]
# yhat = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]
# @test auc("p")(yhat, y) ≈ 2/3
end
true
