module TestLearningNetworks

# using Revise
using Test
using MLJ
using MLJBase
using CategoricalArrays
import Random.seed!
seed!(1234)


@testset "network #1" begin

    N =100
    X = (x1=rand(N), x2=rand(N), x3=rand(N))
    y = 2X.x1  - X.x2 + 0.05*rand(N)
    
    knn_ = KNNRegressor(K=7)
    
    # split the rows:
    allrows = eachindex(y);
    train, valid, test = partition(allrows, 0.7, 0.15);
    @test vcat(train, valid, test) == allrows
    
    Xtrain = selectrows(X, train)
    ytrain = y[train]
    
    Xs = source(Xtrain)
    ys = source(ytrain)
    
    knn1 = machine(knn_, Xs, ys)
    @test_logs (:info, r"Training") fit!(knn1, verbosity=3)
    knn_.K = 5
    @test_logs (:info, r"Training") fit!(knn1, rows=train[1:end-10], verbosity=2)
    @test_logs (:info, r"Training") fit!(knn1, verbosity=2)
    yhat = predict(knn1, Xs)
    yhat(selectrows(X, test))
    @test rms(yhat(selectrows(X, test)), y[test]) < 0.3
    @test MLJ.is_stale(knn1) == false

end

@testset "network #2" begin
    
    N =100
    X = (x1=rand(N),
         x2=rand(N),
         x3=categorical(rand("yn",N)),
         x4=categorical(rand("yn",N)))
    
    y = 2X.x1  - X.x2 + 0.05*rand(N)
    X = source(X)
    y = source(y)
    
    hot = OneHotEncoder()
    hotM = machine(hot, X)
    W = transform(hotM, X)
    knn = KNNRegressor()
    knnM = machine(knn, W, y)
    yhat = predict(knnM, W)
    
    # should get "Training" for both:
    @test_logs (:info, r"^Training") (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat)
    
    # should get "Not retraining" for both:
    @test_logs (:info, r"^Not retraining") (:info, r"^Not retraining") fit!(yhat)
    
    # should get "Updating" for first, "Training" for second:
    hot.drop_last = true
    @test_logs (:info, r"^Updating")  (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat)
    
    # should get "Not retraining" for both:
    @test_logs (:info, r"^Not retraining") (:info, r"^Not retraining") fit!(yhat)
    
    # should get "Not retraining" for first, "Updating for second":
    knn.K = 17
    @test_logs (:info, r"^Not retraining") (:info, r"^Updating") fit!(yhat)
    
    # should get "Training" for both:
    @test_logs (:info, r"^Training")  (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat, rows=1:100)
    
    # should get "Training" for both"
    @test_logs (:info, r"^Training")  (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat)
    
    # should get "Training" for both"
    @test_logs (:info, r"^Training")  (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat, force=true)
    
    forest = EnsembleModel(atom=ConstantRegressor(), n=4)
    forestM = machine(forest, W, y)
    zhat = predict(forestM, W)
    @test_logs (:info, r"^Not") (:info, r"^Train") fit!(zhat)
    forest.n = 6
    @test_logs (:info, r"^Not") (:info, r"^Updating") (:info, r"Build.*length 4") fit!(zhat)

end
    
@testset "network #3" begin
    
    N =100
    X = (x1=rand(N), x2=rand(N), x3=rand(N))
    y = 2X.x1  - X.x2 + 0.05*rand(N)
    
    XX = source(X)
    yy = source(y)
    
    # construct a transformer to standardize the target:
    uscale_ = UnivariateStandardizer()
    uscale = machine(uscale_, yy)
    
    # get the transformed inputs, as if `uscale` were already fit:
    z = transform(uscale, yy)
    
    # construct a transformer to standardize the inputs:
    scale_ = Standardizer() 
    scale = machine(scale_, XX) # no need to fit
    
    # get the transformed inputs, as if `scale` were already fit:
    Xt = transform(scale, XX)
    
    # do nothing to the DataFrame
    Xa = node(identity, Xt)
    
    # choose a learner and make it machine:
    knn_ = KNNRegressor(K=7) # just a container for hyperparameters
    knn = machine(knn_, Xa, z) # no need to fit
    
    # get the predictions, as if `knn` already fit:
    zhat = predict(knn, Xa)
    
    # inverse transform the target:
    yhat = inverse_transform(uscale, zhat)
    
    # fit-through training:
    @test_logs((:info, r"Training"),
               (:info, r"Features standarized: "),
               (:info, r" *:x1"),
               (:info, r" *:x2"),
               (:info, r" *:x3"),
               (:info, r"Training"),
               (:info, r"Training"),
               fit!(yhat, rows=1:50, verbosity=2))
    @test_logs(
        (:info, r"Not retraining"),
        (:info, r"Not retraining"),
        (:info, r"Not retraining"),
        fit!(yhat, rows=1:50, verbosity=1))
    @test_logs(
        (:info, r"Training"),
        (:info, r"Training"),
        (:info, r"Training"),
        fit!(yhat, verbosity=1))
    knn_.K =67
    @test_logs(
        (:info, r"Not retraining"),
        (:info, r"Not retraining"),
        (:info, r"Updating"),
        fit!(yhat, verbosity=1))

end

@testset "overloading methods for AbstractNode" begin
    A  = rand(2,3)
    As = source(A)
    @test MLJ.matrix(MLJ.table(As))() == A
    
    y = rand(4)
    ys = source(y)
    @test vcat(ys, ys)() == vcat(y, y)
    @test hcat(ys, ys)() == hcat(y, y)
end

end

true
