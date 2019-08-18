module TestTasks

# using Revise
using Test
using MLJ
using Random

X=(x=10:10:44, y=1:4, z=collect("abcd"))
task = SupervisedTask(data=X, target=:y, is_probabilistic=true)

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

@testset "Constructors" begin
    df = (x=10:10:44, y=1:4, z=collect("abcd"), w=[1.0, 3.0, missing])
    types = Dict(:x => Continuous, :z => Multiclass, :w => Count)
    task = @test_logs((:warn, r"Missing values encountered"), (:info, r"\n"),
                      supervised(data=df, types=types,
                                 target=:y, ignore=:y, is_probabilistic=false))
    @test scitype_union(task.X.x) <: Continuous
    @test scitype_union(task.X.w) === Union{Count, Missing}
    @test scitype_union(task.y) <: Count
    @test_logs((:info, r"\nis_probabilistic = true"),
               supervised(task.X, task.y, is_probabilistic=true))
end

end # module
true
