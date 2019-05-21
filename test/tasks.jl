module TestTasks

# using Revise
using Test
using MLJ
using Random
using CategoricalArrays

# shuffle!(::SupervisedTask):
X=(x=10:10:44, y=1:4, z=collect("abcd"))
task = @test_logs((:warn, r"An Unknown"), (:info, r"is_probabilistic = true"),
                  SupervisedTask(data=X, target=:y, is_probabilistic=true))

@testset "Shuffling" begin
    task0=deepcopy(task)
    rng = MersenneTwister(1234)
    shuffle!(rng, task0)
    @test task.X != task0.X
    @test task.y != task0.y
    @test MLJ.selectrows(task.X, task0.y) == task0.X

    Random.seed!(1234)
    rng = MersenneTwister(1234)
    task1=shuffle(task)
    shuffle!(rng, task1)
    @test task.X != task1.X
    @test task.y != task1.y
    @test MLJ.selectrows(task.X, task1.y) == task1.X

    Random.seed!(1234)
    task0_a = deepcopy(task)
    shuffle!(task0_a)
    @test task0.X == task0_a.X
    @test task0.y == task0_a.y
    @test MLJ.selectrows(task.X, task0_a.y) == task0_a.X

    Random.seed!(1234)
    rng = MersenneTwister(1234)
    task1_a = shuffle(rng, shuffle(task))
    @test task1.X == task1_a.X
    @test task1.y == task1_a.y
    @test MLJ.selectrows(task.X, task1_a.y) == task1_a.X
end

@testset "Indexing" begin
    task2 = task[:]
    @test count(fieldnames(typeof(task))) do fld
        getproperty(task2, fld) != getproperty(task, fld)
    end == 0
    @test task[2:3].X.z == ['b', 'c']
    @test task[2:3].y == [2, 3]
end

@testset "Type coercion" begin
    types = Dict(:x => Continuous, :z => Multiclass)
    X_coerced = @test_logs coerce(types, task.X)
    @test scitype_union(X_coerced.x) === Continuous
    @test scitype_union(X_coerced.z) <: Multiclass
    @test !X_coerced.z.pool.ordered
    @test_throws MethodError coerce(Count, ["a", "b", "c"])
    y = collect(Float64, 1:5)
    y_coerced = coerce(Count, y)
    @test scitype_union(y_coerced) === Count
    @test y_coerced == y
    y = [1//2, 3//4, 6//5]
    y_coerced = coerce(Continuous, y)
    @test scitype_union(y_coerced) === Continuous
    @test y_coerced ≈ y
    X_coerced = @test_logs coerce(Dict(:z => OrderedFactor), task.X)
    @test X_coerced.x === task.X.x
    @test scitype_union(X_coerced.z) <: OrderedFactor
    @test X_coerced.z.pool.ordered
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
    X_coerced = coerce(Dict(), task.X)
    @test X_coerced.x === task.X.x
    @test X_coerced.z === task.X.z
    z = categorical(task.X.z)
    @test coerce(Multiclass, z) === z
    z = categorical(task.X.z, true, ordered = false)
    @test coerce(Multiclass, z) === z
    z = categorical(task.X.z, true, ordered = true)
    @test coerce(OrderedFactor, z) === z
    # missing values
    y_coerced = @test_logs((:warn, r"Missing values encountered"),
                           coerce(Continuous, [4, 7, missing]))
    @test ismissing(y_coerced == [4.0, 7.0, missing])
    @test scitype_union(y_coerced) === Union{Missing,Continuous}
    y_coerced = @test_logs((:warn, r"Missing values encountered"),
                           coerce(Continuous, Any[4, 7.0, missing]))
    @test ismissing(y_coerced == [4.0, 7.0, missing])
    @test scitype_union(y_coerced) === Union{Missing,Continuous}
    y_coerced = @test_logs((:warn, r"Missing values encountered"),
                           coerce(Count, [4.0, 7.0, missing]))
    @test ismissing(y_coerced == [4, 7, missing])
    @test scitype_union(y_coerced) === Union{Missing,Count}
    y_coerced = @test_logs((:warn, r"Missing values encountered"),
                           coerce(Count, Any[4, 7.0, missing]))
    @test ismissing(y_coerced == [4, 7, missing])
    @test scitype_union(y_coerced) === Union{Missing,Count}
    @test scitype_union(@test_logs((:warn, r"Missing values encountered"),
                                   coerce(Multiclass, [:x, :y, missing]))) ===
        Union{Missing, Multiclass{2}}
    @test scitype_union(@test_logs((:warn, r"Missing values encountered"),
                                   coerce(OrderedFactor, [:x, :y, missing]))) ===
                                       Union{Missing, OrderedFactor{2}}
    # non-missing Any vectors
    @test coerce(Continuous, Any[4, 7]) == [4.0, 7.0]
    @test coerce(Count, Any[4.0, 7.0]) == [4, 7]

    # corner case of using dictionary of types on an abstract vector:
    @test scitype_union(coerce(Dict(:x=>Count), [1.0, 2.0])) <:  Count

end

@testset "Constructors" begin
    df = (x=10:10:44, y=1:4, z=collect("abcd"), w=[1.0, 3.0, missing])
    types = Dict(:x => Continuous, :z => Multiclass, :w => Count)
    task = @test_logs((:warn, r"Missing values encountered"), (:info, r"\n"),
                       supervised(data=df, types=types, target=:y, ignore=:y, is_probabilistic=false))
    @test scitype_union(task.X.x) <: Continuous
    @test scitype_union(task.X.w) === Union{Count, Missing}
    @test scitype_union(task.y) <: Count
    @test_logs((:info, r"\nis_probabilistic = true"),
               supervised(task.X, task.y, is_probabilistic=true))
end

end # module
true
