# simple example on using tasks 

using MLJ
using RDatasets
using DataFrames
using Random

# show all models:
models()

# load some data:
iris = dataset("datasets", "iris");
first(iris, 6)

# shuffle rows:
iris = iris[shuffle(1:size(iris,1)), :];
first(iris, 6)

# define a task:
task = supervised(data=iris, targets=:Species, is_probabilistic=false)

# find models matching the task:
models(task)

task.is_probabilistic = true
models(task)

# load a model:
@load DecisionTreeClassifier
localmodels(task)

# instantiate a model, wrap in a task and evaluate:
tree = DecisionTreeClassifier()
tree.max_depth = 3
mach = machine(tree, task)
evaluate!(mach, resampling=Holdout(fraction_train=0.8), measure=misclassification_rate)

true
