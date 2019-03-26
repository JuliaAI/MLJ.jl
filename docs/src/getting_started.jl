using MLJ
using RDatasets

iris = dataset("datasets", "iris"); # a DataFrame

const X = iris[:, 1:4];
const y = iris[:, 5];

@load DecisionTreeClassifier
tree_model = DecisionTreeClassifier(target_type=String, max_depth=2)

tree = machine(tree_model, X, y)

train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
fit!(tree, rows=train)
yhat = predict(tree, X[test,:]);
misclassification_rate(yhat, y[test]);

evaluate!(tree, resampling=Holdout(fraction_train=0.7, shuffle=true), measure=misclassification_rate)

tree_model.max_depth = 3
evaluate!(tree, resampling=Holdout(fraction_train=0.5, shuffle=true), measure=misclassification_rate)

0.06666666666666667
