using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

#-

using MLJ


# ## Getting some data:

using RDatasets
iris = dataset("datasets", "iris"); # a DataFrame
scrambled = shuffle(1:size(iris, 1))
X = iris[scrambled, 1:4];
y = iris[scrambled, 5];

first(X, 4)

#-

y[1:5]


# ## Basic fit and predict:

@load SVC()
classifier_ = SVC()
classifier = machine(classifier_, X, y)
fit!(classifier)
ŷ = predict(classifier, X) # or some Xnew

#-

# ## Evaluating the model:
evaluate!(classifier,
          resampling=Holdout(fraction_train=0.8),
          measure=misclassification_rate)


# ## Adding dimension reduction:

@load PCA
dim_reducer_ = PCA()
dim_reducer = machine(dim_reducer_, X)
fit!(dim_reducer)
Xsmall = transform(dim_reducer, X);

first(Xsmall, 3)

#-

classifier = machine(classifier_, Xsmall, y)
fit!(classifier)
ŷ = predict(classifier, Xsmall)


# ## Building a composite model:

# ### Method 1: Compact syntax (but not generalizable):

# (not implemented at time of talk)

## composite_ = @pipeline dim_reducer_ classifier_ 

## composite = machine(composite_, X, y)
## evaluate!(composite, measure=misclassification_rate)


# ### Method 2: Re-interpret unstreamlined code:

Xraw = X;
yraw = y;

X = source(Xraw)  
y = source(yraw)

dim_reducer = machine(dim_reducer_, X)
Xsmall = transform(dim_reducer, X)

classifier = machine(classifier_, Xsmall, y)
ŷ = predict(classifier, Xsmall)

#-

fit!(ŷ)

#-

ŷ(rows=3:4)

#-

dim_reducer_.ncomp = 1  # maximum output dimension
fit!(ŷ)

#-

ŷ(rows=3:4)

#  Changing classifier hyperparameter does not retrigger retraining of
#  upstream dimension reducer:

classifier_.gamma = 0.1
fit!(ŷ)

#-

ŷ(rows=3:4)

# Predicting on new data (`Xraw` in `source(Xraw)` is substituted for `Xnew`):

Xnew = (SepalLength = [4.0, 5.2],
        SepalWidth = [3.2, 3.0],
        PetalLength = [1.2, 1.5],
        PetalWidth = [0.1, 0.4],)
ŷ(Xnew)


# #### Exporting network as stand-alone reusable model:

composite_ = @from_network Composite(pca=dim_reducer_, svc=classifier_) <= (X, y, ŷ)
params(composite_)

#-

composite = machine(composite_, Xraw, yraw)
evaluate!(composite, measure=misclassification_rate)

# ## Evaluating a "self-tuning" random forest (nested resampling):

task = load_boston()
models(task)

#-

# ### Evaluating a single tree:

@load DecisionTreeRegressor # load code

tree_ = DecisionTreeRegressor(n_subfeatures=3)
tree = machine(tree_, task)
evaluate!(tree,
          resampling=Holdout(fraction_train=0.7),
          measure=[rms, mav])

# ### Use ensembling wrapper to create a random forest:

forest_ = EnsembleModel(atom=tree_, n=10)


# ### Wrapping in a tuning strategy creates a "self_tuning" random forest:

r1 = range(forest_, :bagging_fraction, lower=0.4, upper=1.0);
r2 = range(forest_, :(atom.n_subfeatures), lower=1, upper=12)

self_tuning_forest_ = TunedModel(model=forest_, 
                          tuning=Grid(),
                          resampling=CV(),
                          ranges=[r1,r2],
                          measure=rms)

# ### Evaluate the self_tuning_forest (nested resampling):

self_tuning_forest = machine(self_tuning_forest_, task)

evaluate!(self_tuning_forest,
          resampling=CV(),
          measure=[rms,rmslp1])



                              

