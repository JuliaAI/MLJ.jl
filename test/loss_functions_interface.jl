# using Revise
using MLJ
using Test
using LossFunctions
import Random.seed!
seed!(1234)

@testset "interface to LossFunctions.jl" begin

    # losses for binary targets:

    y =    categorical(["yes", "yes", "no", "yes"])
    yes, no = y[1], y[3]
    dyes = UnivariateFinite([yes, no], [0.6, 0.4])
    dno =  UnivariateFinite([yes, no], [0.3, 0.7])
    yhat = [dno, dno, dyes, dyes]
    X = nothing
    w = [1, 2, 3, 4]

    @test MLJ.value(ZeroOneLoss(), yhat, X, y, nothing) ≈ [1, 1, 1, 0]
    @test MLJ.value(ZeroOneLoss(), yhat, X, y, w) ≈ [1, 2, 3, 0] ./10 .* 4

    N = 10
    y = categorical(rand(["yes", "no"], N), ordered=true)
    levels!(y, ["no", "yes"])
    no, yes = MLJ.classes(y[1])
    @test MLJ.pm1([yes, no]) == [+1, -1]
    ym = MLJ.pm1(y) # observations for raw LossFunctions measure
    p_vec = rand(N) # probabilities of yes
    yhat  = map(p_vec) do p
        UnivariateFinite([yes, no], [p, 1 - p])
    end
    yhatm = MLJ._scale.(p_vec) # predictions for raw LossFunctions measure
    w = rand(N)
    X = nothing

    for m in [ZeroOneLoss(), L1HingeLoss(), L2HingeLoss(), LogitMarginLoss(),
              ModifiedHuberLoss(), PerceptronLoss(), SmoothedL1HingeLoss(0.9),
              L2MarginLoss(), ExpLoss(), SigmoidLoss(), DWDMarginLoss(0.9)]
        @test MLJ.value(m, yhat, X, y, nothing) ≈ m(yhatm, ym)
        @test mean(MLJ.value(m, yhat, X, y, w)) ≈
            value(m, yhatm, ym, AggMode.WeightedMean(w))
    end

    # losses for continuous targets:

    y  = randn(N)
    yhat = randn(N)
    

    for m in [LPDistLoss(0.5), L1DistLoss(), L2DistLoss(),
              HuberLoss(0.9), EpsilonInsLoss(0.9), L1EpsilonInsLoss(0.9),
              L2EpsilonInsLoss(0.9), LogitDistLoss(), QuantileLoss(0.7)]

        @test MLJ.value(m, yhat, X, y, nothing) ≈ m(yhat, y)
        @test mean(MLJ.value(m, yhat, X, y, w)) ≈
            value(m, yhat, y, AggMode.WeightedMean(w))
    end

end
true

    
