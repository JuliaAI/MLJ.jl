module TestDecisionTree

using MLJ
using Test

task = load_iris();

# get some binary classification data for testing
X, y = X_and_y(task)
X_array = convert(Array{Float64}, X);
 
import DecisionTree
baretree = DecisionTreeClassifier(target_type=String)

# split the rows:
allrows = eachindex(y)
train, test = partition(allrows, 0.7)
@test vcat(train, test) == allrows

baretree.max_depth = 1 
fitresult, cache, report = MLJ.fit(baretree, 1, :, X_array, y);
baretree.max_depth = -1 # no max depth
fitresult, cache, report = MLJ.update(baretree, 1, fitresult, cache, :, X_array, y);

# in this case decision tree is a perfect predictor:
yhat = predict(baretree, fitresult, X_array);
@test yhat == y

# but pruning upsets this:
baretree.post_prune = true
baretree.merge_purity_threshold=0.1
fitresult, cache, report = MLJ.update(baretree, 1, fitresult, cache, :, X_array, y)
yhat = predict(baretree, fitresult, X_array);
@test yhat != y

end
