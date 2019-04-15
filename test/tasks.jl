module TestTasks

# using Revise
using Test
using MLJ
using Random
using CategoricalArrays

# shuffle!(::SupervisedTask):
X=(x=10:10:44, y=1:4, z=collect("abcd"))
task = SupervisedTask(data=X, target=:y, is_probabilistic=true)
task0=deepcopy(task)
rng = MersenneTwister(1234)
shuffle!(rng, task0)
@test task.X != task0.X
@test task.y != task0.y
@test MLJ.selectrows(task.X, task0.y) == task0.X
task1=deepcopy(task)
Random.seed!(1234)
rng = MersenneTwister(1234)
task1=shuffle(task)
shuffle!(rng, task1)
@test task.X != task1.X
@test task.y != task1.y
@test MLJ.selectrows(task.X, task1.y) == task1.X


# task indexing:
task2 = task[:]
@test count(fieldnames(typeof(task))) do fld
    getproperty(task2, fld) != getproperty(task, fld)
end == 0
@test task[2:3].X.z == ['b', 'c']
@test task[2:3].y == [2, 3]

@testset "Type coercion" begin
    types = Dict(:x => MLJ.Continuous, :z => MLJ.Multiclass)
    X_coerced = MLJ.coerce(types, task.X)
    @test X_coerced.x isa Vector{Float64}
    @test X_coerced.z isa CategoricalVector{Char, UInt32}
    @test_throws MethodError MLJ.coerce(Count, ["a", "b", "c"])
    # Check no-op coercion
    y1 = rand(Float64, 5)
    @test MLJ.coerce(MLJ.Continuous, y1) === y1
    y2 = rand(Int, 5)
    @test MLJ.coerce(MLJ.Count, y2) === y2
end

end # module
true
