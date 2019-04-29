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
    import MLJ: coerce
    types = Dict(:x => Continuous, :z => Multiclass)
    X_coerced = coerce(types, task.X)
    @test X_coerced.x isa AbstractVector{Float64}
    @test X_coerced.z isa CategoricalVector{Char, UInt32}
    @test !X_coerced.z.pool.ordered
    @test_throws MethodError coerce(Count, ["a", "b", "c"])
    y = collect(Float64, 1:5)
    y_coerced = coerce(Count, y)
    @test y_coerced isa Vector{Int}
    @test y_coerced == y
    y = [1//2, 3//4, 6//5]
    y_coerced = coerce(Continuous, y)
    @test y_coerced isa Vector{Float64}
    @test y_coerced ≈ y
    y = task.X.z
    y_coerced = coerce(FiniteOrderedFactor, y)
    @test y_coerced isa CategoricalVector{Char, UInt32}
    @test y_coerced.pool.ordered
    # Check no-op coercion
    y = rand(Float64, 5)
    @test coerce(Continuous, y) === y
    y = rand(Float32, 5)
    @test coerce(Continuous, y) === y
    y = rand(BigFloat, 5)
    @test coerce(Continuous, y) === y
    y = rand(Int, 5)
    @test coerce(Count, y) === y
    y = big.(y)
    @test coerce(Count, y) === y
    y = rand(UInt32, 5)
    @test coerce(Count, y) === y
    # missing values
    y_coerced = @test_logs((:warn, r"Missing values encountered"),
                           coerce(Continuous, [4, 7, missing]))
    @test ismissing(y_coerced == [4.0, 7.0, missing])
    @test y_coerced isa Vector{Union{Missing,Float64}}
    y_coerced = @test_logs((:warn, r"Missing values encountered"),
                           coerce(Continuous, Any[4, 7.0, missing]))
    @test ismissing(y_coerced == [4.0, 7.0, missing])
    @test y_coerced isa Vector{Union{Missing,Float64}}
    y_coerced = @test_logs((:warn, r"Missing values encountered"),
                           coerce(Count, [4.0, 7.0, missing]))
    @test ismissing(y_coerced == [4, 7, missing])
    @test y_coerced isa Vector{Union{Missing,Int}}
    y_coerced = @test_logs((:warn, r"Missing values encountered"),
                           coerce(Count, Any[4, 7.0, missing]))
    @test ismissing(y_coerced == [4, 7, missing])
    @test y_coerced isa Vector{Union{Missing,Int}}
    @test eltype(coerce(Multiclass, [:x, :y, missing])) ==
        Union{Missing, CategoricalValue{Symbol,UInt32}}
    @test eltype(coerce(FiniteOrderedFactor, [:x, :y, missing])) ==
        Union{Missing, CategoricalValue{Symbol,UInt32}}
    # non-missing Any vectors
    @test coerce(Continuous, Any[4, 7]) == [4.0, 7.0]
    @test coerce(Count, Any[4.0, 7.0]) == [4, 7]
end

end # module
true
