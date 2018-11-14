module TestKNN

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
fitresult, cache, report = MLJ.fit(knn, 0, X, y); 

r = 1 + 1/sqrt(5) + 1/sqrt(10)
Xtest = [1.0 1.0]
ypred = (1 + 8/sqrt(5) + 2/sqrt(10))/r
@test isapprox(predict(knn, fitresult, Xtest)[1], ypred)

knn.K = 2
fitresult, cache, report = MLJ.update(knn, 0, fitresult, cache, X, y); 
@test predict(knn, fitresult, Xtest)[1] !=  ypred

end
