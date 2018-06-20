include("MLJ.jl")

# TODO:
#   parse wrappers and extract information
#   fix libsvm predict function
#   make the task regression/classification actually useful
#   add more models


# Decision trees example
srand(1)
load("decisiontree")
data = FakedataClassif(1000,3)

task = Task(task_type=:classification, targets=[4], data=data)
lrn = ModelLearner(:forest, Dict(:nsubfeatures=>2, :ntrees=>10))

modelᵧ = learnᵧ(lrn, task)
predictᵧ(modelᵧ, data[:,task.features], task)


# Multivariate example
load("multivariate")
data = Fakedata(1000,4)

task = Task(task_type=:regression, targets=[3], data=data)
lrn = ModelLearner("multivariate", Dict(:regType=>:llsq))

modelᵧ = learnᵧ(lrn, task)
predictᵧ(modelᵧ, data[:, task.features], task)



## Example regression using GLM with penalty and λ tuning
load("glm")
data = Fakedata(1000,3)

ps = ParametersSet([
    ContinuousParameter(
        name = :λ,
        lower = -4,
        upper = -2,
        transform = x->10^x
    ),
    DiscreteParameter(
        name = :penalty,
        values = [L2Penalty(), L1Penalty()]
    )

    ])


task = Task(task_type=:regression, targets=[4], data=data)
lrn = ModelLearner("glm")

storage = MLRStorage()

lrn = tune(lrn, task, ps, measure=mean_squared_error, storage=storage)

include("Visualisation.jl")

plot_storage(storage, plotting_args=Dict(:title=>"A visualisation example"))

# Example classification using SVM with type and cost tuning

load("libsvm")
ps = ParametersSet([
    ContinuousParameter(
        name = :cost,
        lower = -4,
        upper = 1,
        transform = x->10^x
    ),
    DiscreteParameter(
        name = :svmtype,
        values = [SVC()]
    ),
    DiscreteParameter(
        name = :kernel,
        values = [Kernel.Polynomial]
    ),
    ContinuousParameter(
        name = :coef0,
        lower = -4,
        upper = 1,
        transform = x->10^x
    )])

data = FakedataClassif(1000,3)

task = Task(task_type=:classification, targets=[4], data=data)
lrn = ModelLearner("libsvm")

lrn = tune(lrn, task, ps, measure=accuracy)

predictᵧ(lrn, data[:,1:3], task)

# Multiplex example
load("glm")
load("multivariate")

data = Fakedata(1000,4)
task = Task(task_type=:regression, targets=[5], data=data)

lrns = Array{Learner}(0)
psSet = Array{ParametersSet}(0)

lrn = ModelLearner("glm")
ps = ParametersSet([
    ContinuousParameter(
        name = :cost,
        lower = -4,
        upper = 1,
        transform = x->10^x
    ),
    DiscreteParameter(
        name = :penalty,
        values = [L2Penalty(), L1Penalty()]
    )])
push!(lrns, lrn)
push!(psSet, ps)

lrn = ModelLearner("multivariate")
ps = ParametersSet([
    DiscreteParameter(
        name=:regType,
        values = [:llsq, :ridge]
    ),
    ContinuousParameter(
        name = :λ,
        lower = -4,
        upper = 1,
        transform = x->10^x
    )])

push!(lrns, lrn)
push!(psSet, ps)

storage = MLRStorage()
mp = MLRMultiplex(lrns, psSet)

tune(mp, task, storage=storage, measure=mean_squared_error)

plot_storage(storage, plotting_args=Dict(:ylim=>(0.0,0.035)))


# Stacking
lrns = Array{Learner,1}(0)
push!(lrns, ModelLearner("decisiontree", ParametersSet([
    ContinuousParameter(
        name=:maxlabels,
        lower = 1,
        upper = 4,
        transform = x->x
    ),
    ContinuousParameter(
        name=:nsubfeatures,
        lower = 2,
        upper = 3,
        transform = x->x
    ),
    ContinuousParameter(
        name = :maxdepth,
        lower = 3,
        upper = 12,
        transform = x->x
    )])))
push!(lrns, ModelLearner("libsvm", ParametersSet([
    ContinuousParameter(
        name = :cost,
        lower = -4,
        upper = 1,
        transform = x->10^x
    ),
    DiscreteParameter(
        name = :svmtype,
        values = [SVC()]
    )])))
push!(lrns, ModelLearner("libsvm", ParametersSet([
    ContinuousParameter(
        name = :cost,
        lower = -4,
        upper = 1,
        transform = x->10^x
    ),
    DiscreteParameter(
        name = :svmtype,
        values = [NuSVC()]
    )])))

data = FakedataClassif(1000,4)
stacking = CompositeLearner(Stacking(MAJORITY), lrns)
task = Task(task_type=:classification, targets=[5], data=data)
storage = MLRStorage()
tune(stacking, task, storage=storage, measure=accuracy)

pp = predictᵧ(stacking, data[:,task.features], task)

accuracy(pp, data[:,task.targets[1]])


# Different fake data for classification checks
x = zeros(20,2)
y = zeros(Int64, 20,1)
for i in 1:10
    x[i,:] = [rand(), 1]
    y[i] = 0
end

for i in 11:20
    x[i,:] = [rand(), 0]
    y[i] = 1
end
y[3] = 1

using Plots
scatter(x,y, color=y)


data = hcat(x,y)
