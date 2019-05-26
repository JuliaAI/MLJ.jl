module TestKNN

# using Revise
using Test
using MLJ

Xtr = [4 2 5 3;
       2 1 6 0.0];

@test MLJ.KNN.distances_and_indices_of_closest(3,
        MLJ.KNN.euclidean, Xtr, [1, 1])[2] == [2, 4, 1]

X = Xtr' |> collect
y = Float64[2, 1, 3, 8]
knn = KNNRegressor(K=3)
allrows = 1:4
Xtable = MLJ.table(X)
fitresult, cache, report = MLJ.fit(knn, 0, X, y); 

r = 1 + 1/sqrt(5) + 1/sqrt(10)
Xtest = MLJ.table([1.0 1.0])
ypred = (1 + 8/sqrt(5) + 2/sqrt(10))/r
@test predict(knn, fitresult, Xtest)[1] ≈ ypred

knn.K = 2
fitresult, cache, report = MLJ.update(knn, 0, fitresult, cache, X, y); 
@test predict(knn, fitresult, Xtest)[1] !=  ypred

info(knn)

X, y = X_and_y(load_boston())
knnM = machine(knn, X, y)
@test_logs (:info, r"Training") fit!(knnM)
predict(knnM, MLJ.selectrows(X, 1:10))

end
true
