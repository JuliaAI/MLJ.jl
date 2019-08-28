module TestLoading

# using Revise
using Test
using MLJ

@testset "loading of model implementations" begin
    tree = @load DecisionTreeClassifier pkg=DecisionTree verbosity=1
    @test (@isdefined DecisionTreeClassifier)
    @test tree == DecisionTreeClassifier()
    @test_logs((:info, r"^A model"),
               load("DecisionTreeClassifier", mod=TestLoading))
    @test model("DecisionTreeClassifier") in localmodels(mod=TestLoading)
end

end # module

true
