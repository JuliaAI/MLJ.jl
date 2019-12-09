module TestLearningCurves

# using Revise
using Test
using MLJ
import Random.seed!
seed!(1234)

@load KNNRegressor

include("foobarmodel.jl")

x1 = rand(100);
x2 = rand(100);
x3 = rand(100);
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(100);

@testset "learning curves" begin
    atom = FooBarRegressor()
    ensemble = EnsembleModel(atom=atom, n=50, rng=1)
    mach = machine(ensemble, X, y)
    r_lambda = range(ensemble, :(atom.lambda),
                     lower=0.0001, upper=0.1, scale=:log10)
    curve = @test_logs((:info, r"Using measure=rms"),
                       (:info, r"Training"), 
                       (:info, r"Training of best"),
                       MLJ.learning_curve!(mach; range=r_lambda))
    atom.lambda=0.3
    r_n = range(ensemble, :n, lower=10, upper=100)
    curve2 = MLJ.learning_curve!(mach; range=r_n)
    curve3 = learning_curve(ensemble, X, y; range=r_n)
    @test curve2.measurements ≈ curve3.measurements
end

end # module
true
