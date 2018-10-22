include("src/MLJ.jl")
using MLJ
using RDatasets

iris = dataset("datasets", "iris");
target = string.(iris[:Species]);
features = convert(Array, iris[:,1:4]);
labels = string.(iris[:Species]);

# 1. Decision tree classification example
parameters = Dict("pruning_purity" => 1.0
                , "max_depth" => -1
                , "min_samples_leaf" => 1
                , "min_samples_split" => 2
                , "min_purity_increase" => 0.0
                , "n_subfeatures" => 0)

my_model = DecisionTreeClassifier(parameters)
load_interface_for(my_model)
my_tree = fit(my_model, features, target)

result = predict(my_model, my_tree, [5.9,3.0,5.1,1.9])
result

# 2. Decision tree regression example
n, m = 10^3, 5
reg_features = randn(n, m)
weights = rand(-2:2, m)
reg_labels = reg_features * weights

reg_parameters = Dict("pruning_purity" => 1.0
                , "max_depth" => -1
                , "min_samples_leaf" => 1
                , "min_samples_split" => 2
                , "min_purity_increase" => 0.0
                , "n_subfeatures" => 0)

my_reg_model = MLJDecisionTreeRegressor(reg_parameters)
load_interface_for(my_reg_model)
my_reg_tree = fit(my_reg_model, reg_features, reg_labels)
reg_result = predict(my_reg_model, my_reg_tree, [-0.9,3.0,5.1,1.9,0.0])
reg_result