module TestResampling

#  using Revise
using Test
using MLJ
using MLJBase
import MLJModels
import Random.seed!
seed!(1234)

include("foobarmodel.jl")

macro test_parallel(var, ex)
    quote
        $(esc(var)) = false
        $ex
        $(esc(var)) = true
        $ex
    end
end


@testset "resampler as machine" begin
    @test_parallel dopar begin
    global dopar
    N = 50
    X = (x1=rand(N), x2=rand(N), x3=rand(N))
    @show MLJBase.selectrows(X, 1:3)
    y = X.x1 -2X.x2 + 0.05*rand(N)
    end
end


end
true
