module TestPipelines

using MLJ
using Test
using Statistics

@testset "linear_learning_network" begin
    t = MLJ.table
    m = MLJ.matrix
    f = FeatureSelector()
    h = OneHotEncoder()
    k = KNNRegressor()
    u = UnivariateStandardizer()
    c = ConstantClassifier()
    
    X = (x1=[1.0, 2.0, 3.0], x2=categorical([:a, :b, :c]))
    y = [1.0, 2.0, 3.0]
    Xs = source(X)
    ys = source(y, kind=:target)
    
    models = [f, h]
    lin = MLJ.linear_learning_network(Xs, nothing, nothing, nothing,
                                      models...) |> MLJ.tree 
    @test lin.operation == transform
    @test lin.model == h
    @test lin.arg1.operation == transform
    @test lin.arg1.model == f
    @test lin.arg1.arg1.source == Xs
    @test lin.arg1.train_arg1.source == Xs
    @test lin.train_arg1 == lin.arg1
    
    models = [f, k]
    lin = MLJ.linear_learning_network(Xs, ys, nothing, nothing,
                                      models...) |> MLJ.tree
    @test lin.operation == predict
    @test lin.model == k
    @test lin.arg1.operation == transform
    @test lin.arg1.model == f
    @test lin.arg1.arg1.source == Xs
    @test lin.arg1.train_arg1.source == Xs
    @test lin.train_arg1 == lin.arg1
    @test lin.train_arg2.source == ys
    
    models = [m, t]
    lin = MLJ.linear_learning_network(Xs, nothing, nothing, nothing,
                                      models...) |> MLJ.tree
    @test lin.operation == t
    @test lin.model == nothing
    @test lin.arg1.operation == m 
    @test lin.arg1.model == nothing
    @test lin.arg1.arg1.source == Xs
    
    # with learned target transformation:
    models = [f, k]
    lin = MLJ.linear_learning_network(Xs, ys, u, nothing,
                                      models...) |> MLJ.tree
    @test lin.operation == inverse_transform
    @test lin.model == u
    @test lin.arg1.operation == predict
    @test lin.arg1.model == k
    @test lin.arg1.arg1.operation == transform
    @test lin.arg1.arg1.model == f
    @test lin.arg1.arg1.arg1.source == Xs
    @test lin.arg1.arg1.train_arg1.source == Xs
    @test lin.arg1.train_arg1 == lin.arg1.arg1
    @test lin.arg1.train_arg2.operation == transform
    @test lin.arg1.train_arg2.model == u
    @test lin.arg1.train_arg2.arg1.source == ys
    @test lin.arg1.train_arg2.train_arg1.source == ys
    @test lin.train_arg1.source == ys
    
    # with static target transformation:
    models = [f, k]
    lin = MLJ.linear_learning_network(Xs, ys, log, exp, models...) |> MLJ.tree
    @test lin.operation == exp
    @test lin.model == nothing
    @test lin.arg1.operation == predict
    @test lin.arg1.model == k
    @test lin.arg1.arg1.operation == transform
    @test lin.arg1.arg1.model == f
    @test lin.arg1.arg1.arg1.source == Xs
    @test lin.arg1.arg1.train_arg1.source == Xs
    @test lin.arg1.train_arg1 == lin.arg1.arg1
    @test lin.arg1.train_arg2.operation == log
    @test lin.arg1.train_arg2.model == nothing
    @test lin.arg1.train_arg2.arg1.source == ys

    # with supervised model not at end and static target transformation:
    broadcast_mode(v) = mode.(v)
    models = [f, c, broadcast_mode]
    lin = MLJ.linear_learning_network(Xs, ys, log, exp, models...) |> MLJ.tree
    @test lin.operation == exp
    @test lin.model == nothing
    @test lin.arg1.operation == broadcast_mode
    @test lin.arg1.model == nothing
    @test lin.arg1.arg1.operation == predict
    @test lin.arg1.arg1.model == c
    @test lin.arg1.arg1.arg1.operation == transform
    @test lin.arg1.arg1.arg1.model == f
    @test lin.arg1.arg1.arg1.arg1.source == Xs
    @test lin.arg1.arg1.arg1.train_arg1.source == Xs
    @test lin.arg1.arg1.train_arg1 == lin.arg1.arg1.arg1
    @test lin.arg1.arg1.train_arg2.operation == log
    @test lin.arg1.arg1.train_arg2.model == nothing
    @test lin.arg1.arg1.train_arg2.arg1.source == ys

    # with supervised model not at end and with learned target transformation:
    models = [f, c, broadcast_mode]
    lin = MLJ.linear_learning_network(Xs, ys, u, nothing,
                                      models...) |> MLJ.tree
    @test lin.operation == inverse_transform
    @test lin.model == u
    @test lin.arg1.operation == broadcast_mode
    @test lin.arg1.model == nothing
    @test lin.arg1.arg1.operation == predict
    @test lin.arg1.arg1.model == c
    @test lin.arg1.arg1.arg1.operation == transform
    @test lin.arg1.arg1.arg1.model == f
    @test lin.arg1.arg1.arg1.arg1.source == Xs
    @test lin.arg1.arg1.arg1.train_arg1.source == Xs
    @test lin.arg1.arg1.train_arg1 == lin.arg1.arg1.arg1
    @test lin.arg1.arg1.train_arg2.operation == transform
    @test lin.arg1.arg1.train_arg2.model == u
    @test lin.arg1.arg1.train_arg2.arg1.source == ys
    @test lin.arg1.arg1.train_arg2.train_arg1.source == ys
    @test lin.train_arg1.source == ys
    
end

## PIPELINE_PREPROCESS

F = FeatureSelector()
H = OneHotEncoder()
K = KNNRegressor()
C = ConstantClassifier()
U = UnivariateStandardizer()

m = "MLJ.matrix"
t = "MLJ.table"
f = "fea=F"
h = "hot=H"
k = "knn=K"
c = "cnst=C"
e = "target=exp"
l = "inverse=log"
u = "target=U"

function pipe(args...)
    ret = string("Pipe(", args[1])
    for x in args[2:end]
        ret *= string(", ", x)
    end
    return Meta.parse(ret*")")
end
MLJ.pipeline_preprocess(modl, ex) =
    MLJ.pipeline_preprocess(modl, ex, missing)

ex = pipe(f, m, t, h, k, u)
@test MLJ.pipeline_preprocess(TestPipelines, ex) ==
    (:Pipe, [:fea, :hot, :knn], [:F, :H, :K],
     [F, MLJ.matrix, MLJ.table, H, K], U, nothing, :DeterministicNetwork)

ex =  pipe(f, m, t, h, k, e, l)
@test MLJ.pipeline_preprocess(TestPipelines, ex) ==
    (:Pipe, [:fea, :hot, :knn], [:F, :H, :K],
     [F, MLJ.matrix, MLJ.table, H, K], exp, log, :DeterministicNetwork)

ex =  pipe(f, k)
@test MLJ.pipeline_preprocess(TestPipelines, ex) ==
    (:Pipe, [:fea, :knn], [:F, :K],
     [F, K], nothing, nothing, :DeterministicNetwork)

ex =  pipe(f, c)
@test MLJ.pipeline_preprocess(TestPipelines, ex, :(is_probabilistic=true)) ==
    (:Pipe, [:fea, :cnst], [:F, :C],
     [F, C], nothing, nothing, :ProbabilisticNetwork)

ex =  pipe(f, h)
@test MLJ.pipeline_preprocess(TestPipelines, ex) ==
    (:Pipe, [:fea, :hot], [:F, :H],
     [F, H], nothing, nothing, :UnsupervisedNetwork)

ex =  :(Pipe())
@test_throws ArgumentError MLJ.pipeline_preprocess(TestPipelines, ex)

# target is a function but no inverse=...:
ex =  pipe(f, m, t, h, k, e)
@test_throws ArgumentError MLJ.pipeline_preprocess(TestPipelines, ex)

# inverse but no target:
ex =  pipe(f, k, l)
@test_throws ArgumentError MLJ.pipeline_preprocess(TestPipelines, ex)

# target specified but no component is supervised:
ex =  pipe(f, h, u)
@test_throws ArgumentError MLJ.pipeline_preprocess(TestPipelines, ex)

# function as target but no inverse:
ex =  pipe(f, m, t, h, k, e)
@test_throws ArgumentError MLJ.pipeline_preprocess(TestPipelines, ex)

# is_probabilistic=true declared but no supervised models
ex = pipe(f, m, t, h)
@test_throws ArgumentError MLJ.pipeline_preprocess(TestPipelines, ex,
                                                   :(is_probabilistic=true))


end
true
