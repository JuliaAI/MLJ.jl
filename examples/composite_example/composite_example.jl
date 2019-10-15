# # A composite model example with column splits and merges

# The following composite model description comes from a
# discussion at [MLJ issue #166](https://github.com/alan-turing-institute/MLJ.jl/issues/166#issuecomment-533934909):

# > Regress y from x, and classify c from a and b. Then classify w
# > from y and c.

# Below we show how to use MLJ to define a new supervised model type
# `MyComposite` with input (a, b, x) to learn a target (c, y, w)
# according to this prescription. The fields (hyperparmeters) of the
# new composite model will be the two classifiers and regresssor.

# The new model type is obtained by "protyping" the composite model
# using a learning network, and then exporting the network as a
# stand-alone model type.

# Select the relevant MLJ version in the [manual
# entry](https://alan-turing-institute.github.io/MLJ.jl/stable/composing_models/)
# for more on this general procedure.

# To run without issues, this notebook/script should lie in a copy of
# [this
# directory](https://github.com/alan-turing-institute/MLJ.jl/tree/master/examples/composite_example),
# in some tagged release of the [MLJ
# package](https://github.com/alan-turing-institute/MLJ.jl).

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

#-

using MLJ
using Random
Random.seed!(12);


# ### ASSUMPTIONS

# We will make the following assumptions regarding the scientific types
# of the data:

# - `a`, `b`, `x` have scitype  `AbstractMatrix{Continuous}`

# - `c` and `w` have scitype `AbstractVector{<:Finite}`

# - `y` has scitype `AbstractVector{Continuous}`

# All data share the same number of rows (corresponding to
# observations).

# For example,

N = 2
a = fill(1.0, (3N, 3)) + rand(3N, 3)
b = fill(2.0, (3N, 2)) + rand(3N, 2);
x = fill(3.0, (3N, 2)) + rand(3N, 2);
c = categorical(rand("pqr", 3N)); levels!(c, ['p', 'q', 'r'])
w = categorical(rand("PQ", 3N)); levels!(w, ['P', 'Q'])
y = fill(4.0, 3N) + rand(3N);

# I'll suppose the input to our supervised composite model is to be
# presented as a matrix of the form `X = hcat(a, b, x)` where `a`,
# `b`, `x` are of the form above.  For example,

X = hcat(a, b, x);

scitype(X)

# Since the three target variables `c`, `y`, `z` for the composite
# have different types, I'll suppose that these are presented as the
# three columns of a table, with names :c, :y, and :z. For example,

Y = (c=c, y=y, w=w);

scitype(Y)

# We are assuming the learners are:

# - A probabilisitic classifier for learning c from a, b
# - A deterministic regressor for learning y from x
# - A deterministic classifier for learning w from c and y

# Here "classifier" means `AbstractVector{<:Finite}` target scitype,
# and "regressor" means `AbstractVector{<:Continuous}` target scitype.

# We restrict to component models that have `Table(Continous)` input
# scitype and so will need to one-hot encode (c, y), before learning
# w.


# ### PROTYPING THE COMPOSITE MODEL

# Now we define a learning network using component models that will
# become default values for the fields (hyperparameters) of our final
# composite model type.

# We first define ordinary functions to do splitting and merging. The
# functions return a table or vector depending on what the component
# models will be requiring (in this case, tables for inputs, vectors
# for targets):


# Splits:
get_ab(X) = MLJ.table(X[:,1:5], names=[:a1, :a2, :a3, :b1, :b2])
get_x(X)  = MLJ.table(X[:,6:7], names=[:x1, :x2])
get_c(Y)  = Y.c
get_y(Y)  = Y.y
get_w(Y)  = Y.w;

# Merges:
put_cy(c, y) = (c=c, y=y)
put_cyw(c, y, w) = (c=c, y=y, w=w);

#-

get_ab(X) |> pretty
get_x(X) |> pretty
put_cy(c, y) |> pretty
put_cyw(c, y, w) |> pretty

# We now define source nodes. These nodes could simply wrap `nothing`
# instead of concrete data, and the network could still be exported.
# However, to enable testing of the learning network as we build it,
# we will wrap the data defined above. (The author discovered several
# errors in earlier attempts this way.)

X_ = source(X)
Y_ = source(Y, kind=:target)

# Now for the rest of the network.

# Initial splits:
ab_ = node(get_ab, X_)
x_ = node(get_x, X_)
c_ = node(get_c, Y_)
y_ = node(get_y, Y_)
w_ = node(get_w, Y_)

# Node to predict c:
clf1 = @load DecisionTreeClassifier # a model instance
m1 = machine(clf1, ab_, c_)
ĉ_ = predict_mode(m1, ab_)

# Node to predict y:
rgs = @load RidgeRegressor pkg=MultivariateStats
rgs.lambda = 0.1
m = machine(rgs, x_, y_)
ŷ_ = predict(m, x_)

# Merge c and y:
cy_ = node(put_cy, ĉ_, ŷ_)

# Node to do the one-hot-encoding:
hot = OneHotEncoder(drop_last=true)
cy__ = transform(machine(hot, cy_), cy_)

# Node to predict w:
clf2 = @load SVC
m2 = machine(clf2, cy__, w_)
ŵ_ = predict(m2, cy__)

# Final merge:
Ŷ_ = node(put_cyw, ĉ_, ŷ_, ŵ_)

# As a test of functionality, we can fit the final node, which trains
# the whole network...

fit!(Ŷ_, rows=1:2N)

# ... and make a prediction:

Ŷ_(rows=(2N-1):3N)


# ### EXPORT THE LEARNING NETWORK AS STAND-ALONE MODEL

# The next code simultaneously creates a new model type `MyComposite`
# and defines `comp` as an instance, using deep copies of the
# specified learning network component models as default field values:

comp = @from_network MyComposite(classifier1=clf1,
                                 classifier2=clf2,
                                 regressor=rgs) <= Ŷ_

# As a model, this object has no data attached to it. We fit it to
# data, as we do any other model:

X = rand(100, 7);
Y = (c=categorical(rand("abc", 100)),
     y=rand(100),
     w=categorical(rand("AB", 100)));

m = machine(comp, X, Y)
fit!(m, rows=1:80)
Ŷ = predict(m, rows=81:100)
error = sum(Ŷ.w .!= Y.w[81:100])/20

# We can select new component models, for example ...

comp.classifier1 = @load KNNClassifier

# ... and retrain:

fit!(m, rows=1:80)
Ŷ = predict(m, rows=81:100)
error = sum(Ŷ.w .!= Y.w[81:100])/20

using Literate #src
Literate.notebook(@__FILE__, @__DIR__) #src
