module TestPipelines

using MLJ
using Test
using Statistics
import Random.seed!
seed!(1234)

NN = 7
X = MLJ.table(rand(NN, 3));
y = 2X.x1 - X.x2 + 0.05*rand(NN);
Xs = source(X); ys = source(y, kind=:target)

broadcast_mode(v) = mode.(v)

@testset "linear_learning_network" begin
    t = MLJ.table
    m = MLJ.matrix
    f = FeatureSelector()
    h = OneHotEncoder()
    k = KNNRegressor()
    u = UnivariateStandardizer()
    c = ConstantClassifier()
        
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
    lin = MLJ.linear_learning_network(Xs, ys, StaticTransformer(log),
                                      StaticTransformer(exp),
                                      models...) |> MLJ.tree
    @test lin.operation == transform
    @test lin.model.f == exp
    @test lin.arg1.operation == predict
    @test lin.arg1.model == k
    @test lin.arg1.arg1.operation == transform
    @test lin.arg1.arg1.model == f
    @test lin.arg1.arg1.arg1.source == Xs
    @test lin.arg1.arg1.train_arg1.source == Xs
    @test lin.arg1.train_arg1 == lin.arg1.arg1
    @test lin.arg1.train_arg2.operation == transform
    @test lin.arg1.train_arg2.model.f == log 
    @test lin.arg1.train_arg2.arg1.source == ys

    # with supervised model not at end and static target transformation:
    models = [f, c, broadcast_mode]
    lin = MLJ.linear_learning_network(Xs, ys, StaticTransformer(log),
                                      StaticTransformer(exp),
                                      models...) |> MLJ.tree
    @test lin.operation == transform
    @test lin.model.f == exp
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
    @test lin.arg1.arg1.train_arg2.model.f == log
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
    
    # build a linear network for training:
    N = MLJ.linear_learning_network(Xs, ys, u, nothing, f, k)
    fM = machine(f, Xs)
    Xt = transform(fM, Xs)

    # build the same network by hand:
    uM = machine(u, ys)         
    yt = transform(uM, ys)
    kM = machine(k, Xt, yt)
    zhat = predict(kM, Xt)
    N2 = inverse_transform(uM, zhat)

    # compare predictions
    fit!(N); fit!(N2)
    yhat =  N();
    @test yhat ≈ N2()
    k.K = 3; f.features = [:x3,]
    fit!(N); fit!(N2)
    @test !(yhat ≈ N()) 
    @test N() ≈ N2()
    global hand_built = N(); 

end


## PIPELINE_PREPROCESS

F = FeatureSelector()
H = OneHotEncoder()
K = KNNRegressor()
C = ConstantClassifier()
U = UnivariateStandardizer()

m = :(MLJ.matrix)
t = :(MLJ.table)
f = :(fea=F)
h = :(hot=H)
k = :(knn=K)
c = :(cnst=C)
e = :(target=exp)
l = :(inverse=log)
u = :(target=U)

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
out = MLJ.pipeline_preprocess(TestPipelines, ex)
@test out[1] == :Pipe
@test out[2] == [:fea, :hot, :knn, :target]
@test eval.(out[3]) == [F, H, K, U]
@test eval.(out[4]) == [F, MLJ.matrix, MLJ.table, H, K]
@test eval.(out[5]) == U
@test eval.(out[6]) == nothing
@test out[7] == :DeterministicNetwork

ex =  pipe(f, m, t, h, k, e, l)
out = MLJ.pipeline_preprocess(TestPipelines, ex)
@test out[1] == :Pipe
@test out[2] == [:fea, :hot, :knn, :target, :inverse]
@test eval.(out[3])[1:3] == [F, H, K]
@test [eval.(out[3])[4:5]...]  isa AbstractVector{<:StaticTransformer}
@test eval.(out[4]) == [F, MLJ.matrix, MLJ.table, H, K]
@test eval.(out[5]).f == exp
@test eval.(out[6]).f == log
@test out[7] == :DeterministicNetwork

ex =  pipe(f, k)
out = MLJ.pipeline_preprocess(TestPipelines, ex)
@test out[1] == :Pipe
@test out[2] == [:fea, :knn]
@test eval.(out[3]) == [F, K]
@test eval.(out[4]) == [F, K]
@test eval.(out[5]) == nothing
@test eval.(out[6]) == nothing
@test out[7] == :DeterministicNetwork

ex =  pipe(f, c)
out = MLJ.pipeline_preprocess(TestPipelines, ex, :(is_probabilistic=true))
@test out[1] == :Pipe
@test out[2] == [:fea, :cnst]
@test eval.(out[3]) == [F, C]
@test eval.(out[4]) == [F, C]
@test eval.(out[5]) == nothing
@test eval.(out[6]) == nothing
@test out[7] == :ProbabilisticNetwork

ex =  pipe(f, h)
out = MLJ.pipeline_preprocess(TestPipelines, ex)
@test out[1] == :Pipe
@test out[2] == [:fea, :hot]
@test eval.(out[3]) == [F, H]
@test eval.(out[4]) == [F, H]
@test eval.(out[5]) == nothing
@test eval.(out[6]) == nothing
@test out[7] == :UnsupervisedNetwork

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

# test a simple pipeline prediction agrees with prediction of
# hand-built learning network built earlier:
p = @pipeline(Pipe(sel=FeatureSelector(), knn=KNNRegressor(),
                   target=UnivariateStandardizer()))
p.knn.K = 3; p.sel.features = [:x3,]
mach = machine(p, X, y)
fit!(mach)
@test MLJ.tree(mach.fitresult).arg1.model.K == 3
MLJ.tree(mach.fitresult).arg1.arg1.model.features == [:x3, ]
@test predict(mach) ≈ hand_built

X = MLJ.table(rand(7,3))
y = categorical(collect("ffmmfmf"))
Xs = source(X)
ys = source(y, kind=:target)
p = @pipeline(Piper(hot=OneHotEncoder(), cnst=ConstantClassifier(),
                    broadcast_mode))
mach = machine(p, X, y)
fit!(mach)
@test predict(mach) == fill('f', 7) 

end
true

