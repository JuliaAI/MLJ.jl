# # Code used in "Using MLJ. Lesson 1: Basics"

# Ensure your current directory contains this file and change `@__DIR__` to `pwd()` below
# to activate a Julia package environment that has been tested with this notebook:

using Pkg
Pkg.activate(@__DIR__)
# Pkg.activate(pwd())
Pkg.instantiate()

# Presentation starts with evaluation of this cell:

Pkg.status()

# # PART 1. REGRESSIONS

using MLJ
using UnicodePlots

# load data and inspect the schema:
data = load_boston()
schema(data) # 'MedV'= median value of owner-occupied homes in $1000s.

# split off target variable:
y, X = unpack(data, ==(:MedV))
schema(X)
first(y, 5)

# split observations (row indices) in ration 60:40
train, test = partition(1:length(y), 0.6)

# choose a model:
models(matching(X, y))
Regressor = @iload RandomForestRegressor pkg=DecisionTree
model = Regressor()
@doc Regressor

# train on `train` rows:
mach = machine(model, X, y)
fit!(mach, rows=train)

# inspect the machine:
fitted_params(mach)
report(mach)

# predict on some "new" data
predict(mach, X)

# predict in the `test` rows:
ypred = predict(mach, rows=test)

# get the mean absolute error:
mae(ypred, y[test])

# other options for the metric:
mae
measures(ypred, y)

# get performance estimates in one hit:
evaluate!(mach; resampling=[(train, test),], measures=[mae, RSquared()])

# something fancier:
evaluate(model, X, y;
         resampling=CV(nfolds=10),
         measures=[mae, RSquared()],
         )
# Dietterich's 5 x 2 test:
evaluate(
    model, X, y;
    resampling=CV(nfolds=2, shuffle=true),
    repeats=5,
    measures=[mae, RSquared()],
    acceleration=CPUThreads(),
)


# # BREAK OFF FOR SCITYPE DICUSSSION

typeof(3.14)
scitype(3.14)
scitype(3.143f0)
scitype(["cat", "mouse", "dog"])


# # PART 2. CLASSIFICATION

# new data set for classification, the Adult Dataset (census data):
using Downloads, CSV
url = "https://raw.githubusercontent.com/"*
    "saravrajavelu/Adult-Income-Analysis/refs/heads/master/"*
    "adult.csv"
file = Downloads.download(url)
data = CSV.read(file, NamedTuple)
schema(data)

data.income

# fix some of the incorrect scitypes:
data = coerce(data,
              :age=>Continuous,
              :occupation=>Multiclass,
              :gender=>Multiclass,
              Symbol("educational-num")=>OrderedFactor,
              :income=>Multiclass,
);

# split off target, dump some features, and shuffle the observations:
y, X = unpack(
    data,
    ==(:income),
    in([:age, :occupation, :gender, Symbol("educational-num")]),
    rng = 123,
);
scitype(y)

# split observations (row indices) in ration 60:40
train, test = partition(1:length(y), 0.6)

# what models are available?
models(matching(X, y))

# one hot encoding:
model_hot = OneHotEncoder()
localmodels()
mach = machine(model_hot, X) |> fit!
Xhot = transform(mach, X)
schema(Xhot)

# choose a supervised model:
models(matching(Xhot,y))
model = (@load RandomForestClassifier pkg=DecisionTree)()

# evaluate by hand:
mach = machine(model, Xhot, y)
fit!(mach, rows=train)
yprob = predict(mach, rows=test);
first(yprob, 5)
first(yprob)
ypoint = mode.(yprob)
ypoint = predict_mode(mach, rows=test) # same thing
accuracy(ypoint, y[test])
log_loss(yprob, y[test])

# shortcut:
evaluate(model, Xhot, y;
         resampling=Holdout(fraction_train=0.6),
         measures = [accuracy, log_loss],
         )


