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
fitresult, cache, report = MLJ.fit(baretree, 1, X_array, y);
baretree.max_depth = -1 # no max depth
fitresult, cache, report = MLJ.fit2(baretree, 1, fitresult, cache, X_array, y);
fitresult, cache, report = MLJ.fit2(baretree, 1, fitresult, cache, X_array, y
                                    ; prune_only=true, merge_purity_threshold=0.1);


# in this case decision tree is perfect predictor:
fitresult, cache, report = MLJ.fit2(baretree, 1, fitresult, cache, X_array, y);
yhat = predict(baretree, fitresult, X_array);
@test yhat == y

# but pruning upsets this:
fitresult, cache, report = MLJ.fit2(baretree, 1, fitresult, cache, X_array, y
                                    ; prune_only=true, merge_purity_threshold=0.1);
yhat = predict(baretree, fitresult, X_array);
@test yhat != y

end
