module TestDecisionTree

# using Revise
using MLJ
using Test

task = load_iris();

# get some binary classification data for testing
X, y = X_and_y(task)
# X_array = convert(Matrix{Float64}, X);
 
import DecisionTree
import CategoricalArrays
baretree = DecisionTreeClassifier(target_type=String)

# split the rows:
allrows = eachindex(y)
train, test = partition(allrows, 0.7)
@test vcat(train, test) == allrows

baretree.max_depth = 1 
fitresult, cache, report = MLJ.fit(baretree, 1, X, y);
baretree.max_depth = -1 # no max depth
fitresult, cache, report = MLJ.update(baretree, 1, fitresult, cache, X, y);

# in this case decision tree is a perfect predictor:
yhat = predict(baretree, fitresult, X);
@test yhat == y

# but pruning upsets this:
baretree.post_prune = true
baretree.merge_purity_threshold=0.1
fitresult, cache, report = MLJ.update(baretree, 2, fitresult, cache, X, y)
yhat = predict(baretree, fitresult, X);
@test yhat != y

tree = machine(baretree, X, y)
fit!(tree)
predict(tree, MLJ.selectrows(X, 1:3))

@show baretree
info(baretree)

end
true
