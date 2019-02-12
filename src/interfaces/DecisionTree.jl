#> This code implements the MLJ model interface for models in the
#> DecisionTree.jl package. It is annotated so that it may serve as a
#> template for other supervised models of type `Probabilistic`. The
#> annotations, which begin with "#>", should be removed (but copy
#> this file first!). See also the model interface specification at
#> "doc/adding_new_models.md".

#> Note that all models need to "register" their location by setting
#> `load_path(<:ModelType)` appropriately.

module DecisionTree_

#> export the new models you're going to define (and nothing else):
export DecisionTreeClassifier, DecisionTreeRegressor

import MLJBase

#> needed for all classifiers:
using CategoricalArrays

#> import package:
import ..DecisionTree # strange syntax b/s we are lazy-loading

# here T is target type, and the `Vector{T}` is for storing target levels:  
const DecisionTreeClassifierFitResultType{T} =
    Tuple{Union{DecisionTree.Node{Float64,T}, DecisionTree.Leaf{T}}, Vector{T}}

"""
    DecisionTreeClassifer(; kwargs...)

CART decision tree classifier from
[https://github.com/bensadeghi/DecisionTree.jl/blob/master/README.md](https://github.com/bensadeghi/DecisionTree.jl/blob/master/README.md). Predictions
are probabilistic. 

For post-fit pruning, set `post-prune=true` and set
`min_purity_threshold` appropriately. Other hyperparameters as per
package documentation cited above.

"""
mutable struct DecisionTreeClassifier{T} <: MLJBase.Probabilistic{DecisionTreeClassifierFitResultType{T}}
    target_type::Type{T}  # target is CategoricalArray{target_type}
    pruning_purity::Float64
    max_depth::Int
    min_samples_leaf::Int
    min_samples_split::Int
    min_purity_increase::Float64
    n_subfeatures::Float64
    display_depth::Int
    post_prune::Bool
    merge_purity_threshold::Float64
end

# constructor:
#> all arguments are kwargs with a default value
function DecisionTreeClassifier(
    ; target_type=Int
    , pruning_purity=1.0
    , max_depth=-1
    , min_samples_leaf=1
    , min_samples_split=2
    , min_purity_increase=0.0
    , n_subfeatures=0
    , display_depth=5
    , post_prune=false
    , merge_purity_threshold=0.9)

    model = DecisionTreeClassifier{target_type}(
        target_type
        , pruning_purity
        , max_depth
        , min_samples_leaf
        , min_samples_split
        , min_purity_increase
        , n_subfeatures
        , display_depth
        , post_prune
        , merge_purity_threshold)

    message = MLJBase.clean!(model)       #> future proof by including these
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

#> The following optional method (the fallback does nothing, returns
#> empty warning) is called by the constructor above but also by the
#> fit methods below:
function MLJBase.clean!(model::DecisionTreeClassifier)
    warning = ""
    if  model.pruning_purity > 1
        warning *= "Need pruning_purity < 1. Resetting pruning_purity=1.0.\n"
        model.pruning_purity = 1.0
    end
    if model.min_samples_split < 2
        warning *= "Need min_samples_split < 2. Resetting min_samples_slit=2.\n"
        model.min_samples_split = 2
    end
    return warning
end

#> A required `fit` method returns `fitresult, cache, report`. (Return
#> `cache=nothing` unless you are overloading `update`)
function MLJBase.fit(model::DecisionTreeClassifier{T2}
             , verbosity::Int   #> must be here (and typed!!) even if not used (as here)
             , X
             , y::CategoricalVector{T}) where {T,T2}

    T == T2 || throw(ErrorException("Type, $T, of target incompatible "*
                                    "with type, $T2, of $model."))

    Xmatrix = MLJBase.matrix(X)

    classes = levels(y) # *all* levels in pool of y, not just observed ones
    decoder = MLJBase.CategoricalDecoder(y)
    y_plain = MLJBase.transform(decoder, y)

    tree = DecisionTree.build_tree(y_plain
                                   , Xmatrix
                                   , model.n_subfeatures
                                   , model.max_depth
                                   , model.min_samples_leaf
                                   , model.min_samples_split
                                   , model.min_purity_increase)
    if model.post_prune
        tree = DecisionTree.prune_tree(tree, model.merge_purity_threshold)
    end

    verbosity < 3 || DecisionTree.print_tree(tree, model.display_depth)

    fitresult = (tree, classes)

    #> return package-specific statistics (eg, feature rankings,
    #> internal estimates of generalization error) in `report`, which
    #> should be `nothing` or a dictionary keyed on symbols.

    cache = nothing
    report = nothing

    return fitresult, cache, report

end

function MLJBase.predict(model::DecisionTreeClassifier{T}
                     , fitresult
                     , Xnew) where T
    Xmatrix = MLJBase.matrix(Xnew)
    tree, classes = fitresult

    # apply_tree_proba returns zero probabilities on levels unseen in
    # train, so we can give it all the levels in the pool of the
    # training vector:
    y_probabilities = DecisionTree.apply_tree_proba(tree, Xmatrix, classes)
    return [MLJBase.UnivariateNominal(classes, y_probabilities[i,:])
            for i in 1:size(y_probabilities, 1)]
end

# metadata:
MLJBase.load_path(::Type{<:DecisionTreeClassifier}) = "MLJModels.DecisionTree_.DecisionTreeClassifier" 
MLJBase.package_name(::Type{<:DecisionTreeClassifier}) = "DecisionTree"
MLJBase.package_uuid(::Type{<:DecisionTreeClassifier}) = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
MLJBase.package_url(::Type{<:DecisionTreeClassifier}) = "https://github.com/bensadeghi/DecisionTree.jl"
MLJBase.is_pure_julia(::Type{<:DecisionTreeClassifier}) = :yes
MLJBase.input_kinds(::Type{<:DecisionTreeClassifier}) = [:continuous, ]
MLJBase.output_kind(::Type{<:DecisionTreeClassifier}) = :multiclass
MLJBase.output_quantity(::Type{<:DecisionTreeClassifier}) = :univariate

end # module

