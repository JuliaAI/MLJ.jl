module TestLoading

# using Revise
using Test
using MLJ

@testset "loading of model implementations" begin
    @load DecisionTreeClassifier pkg=DecisionTree verbosity=1
    @test (@isdefined DecisionTreeClassifier)
    @test_logs((:info, r"^A model named"),
               load_implementation("DecisionTreeClassifier", mod=TestLoading))
end

end # module

true
