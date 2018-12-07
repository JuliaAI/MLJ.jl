module XGBoost_

#> export the new models you're going to define (and nothing else):
export XGBoostRegressor, XGBoostClassifier

#> for all Supervised models:
import MLJ
import MLJ: CanWeightTarget, CanRankFeatures
import MLJ: Nominal, Numeric, NA, Probababilistic, Multivariate,  Multiclass

#> for all classifiers:
using CategoricalArrays

#> import package:
import XGBoost

mutable struct XGBoostRegressor{Integer} <:MLJ.Supervised{Integer}  #check the parametric type here..
    num_round::Integer
    booster::String
    silent::Union{Int,Bool}
    disable_default_eval_metric::Real
    eta::Real
    gamma::Real
    max_depth::Real
    min_child_weight::Real
    max_delta_step::Real
    subsample::Real
    colsample_bytree::Real
    colsample_bylevel::Real
    lambda::Real
    alpha::Real
    tree_method::String
    sketch_eps::Real
    scale_pos_weight::Real
    updater::String
    refresh_leaf::Union{Int,Bool}
    process_type::String
    grow_policy::String
    max_leaves::Int
    max_bin::Int
    predictor::String
    sample_type::String
    normalize_type::String
    rate_drop::Real
    one_drop
    skip_drop::Real
    feature_selector::String
    top_k::Real
    tweedie_variance_power::Real
    objective
    base_score::Real
    eval_metric
    seed::Integer
    watchlist
    num_class::Integer
end


"""
# constructor:
A full list of the kwargs accepted, and their value ranges, consult
https://xgboost.readthedocs.io/en/latest/parameter.html.
The only required kwarg is num_round.
"""
function XGBoostRegressor(
    ;num_round=1
    ,booster="gbtree"
    ,silent=0  #> might be redundant due to verbosity
    ,disable_default_eval_metric=0
    ,eta=0.3
    ,gamma=0
    ,max_depth=6
    ,min_child_weight=1
    ,max_delta_step=0
    ,subsample=1
    ,colsample_bytree=1
    ,colsample_bylevel=1
    ,lambda=1
    ,alpha=0
    ,tree_method="auto"
    ,sketch_eps=0.03
    ,scale_pos_weight=1
    ,updater="grow_colmaker"
    ,refresh_leaf=1
    ,process_type="default"
    ,grow_policy="depthwise"
    ,max_leaves=0
    ,max_bin=256
    ,predictor="cpu_predictor" #> gpu version not currently working with Julia, maybe remove completely?
    ,sample_type="uniform"
    ,normalize_type="tree"
    ,rate_drop=0.0
    ,one_drop=0
    ,skip_drop=0.0
    ,feature_selector="cyclic"
    ,top_k=0
    ,tweedie_variance_power=1.5
    ,objective="reg:linear"
    ,base_score=0.5
    ,eval_metric="rmse"
    ,seed=0
    ,watchlist=[]
    ,num_class=1)

    model = XGBoostRegressor{Integer}(
    num_round
    ,booster
    ,silent  #> might be redundant due to verbosity
    ,disable_default_eval_metric
    ,eta
    ,gamma
    ,max_depth
    ,min_child_weight
    ,max_delta_step
    ,subsample
    ,colsample_bytree
    ,colsample_bylevel
    ,lambda
    ,alpha
    ,tree_method
    ,sketch_eps
    ,scale_pos_weight
    ,updater
    ,refresh_leaf
    ,process_type
    ,grow_policy
    ,max_leaves
    ,max_bin
    ,predictor #> gpu version not currently working with Julia, maybe remove completely?
    ,sample_type
    ,normalize_type
    ,rate_drop
    ,one_drop
    ,skip_drop
    ,feature_selector
    ,top_k
    ,tweedie_variance_power
    ,objective
    ,base_score
    ,eval_metric
    ,seed
    ,watchlist
    ,num_class)

     message = MLJ.clean!(model)           #> future proof by including these
     isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

mutable struct XGBoostClassifier{Integer} <:MLJ.Supervised{Integer}  #check the parametric type here..
    num_round::Integer
    booster::String
    silent::Union{Int,Bool}
    disable_default_eval_metric::Real
    eta::Real
    gamma::Real
    max_depth::Real
    min_child_weight::Real
    max_delta_step::Real
    subsample::Real
    colsample_bytree::Real
    colsample_bylevel::Real
    lambda::Real
    alpha::Real
    tree_method::String
    sketch_eps::Real
    scale_pos_weight::Real
    updater::String
    refresh_leaf::Union{Int,Bool}
    process_type::String
    grow_policy::String
    max_leaves::Int
    max_bin::Int
    predictor::String
    sample_type::String
    normalize_type::String
    rate_drop::Real
    one_drop
    skip_drop::Real
    feature_selector::String
    top_k::Real
    tweedie_variance_power::Real
    objective
    base_score::Real
    eval_metric
    seed::Integer
    watchlist
    num_class::Integer
end


"""
# constructor:
A full list of the kwargs accepted, and their value ranges, consult
https://xgboost.readthedocs.io/en/latest/parameter.html.
The only required kwarg is num_round.
"""
function XGBoostClassifier(
    ;num_round=1
    ,booster="gbtree"
    ,silent=0  #> might be redundant due to verbosity
    ,disable_default_eval_metric=0
    ,eta=0.3
    ,gamma=0
    ,max_depth=6
    ,min_child_weight=1
    ,max_delta_step=0
    ,subsample=1
    ,colsample_bytree=1
    ,colsample_bylevel=1
    ,lambda=1
    ,alpha=0
    ,tree_method="auto"
    ,sketch_eps=0.03
    ,scale_pos_weight=1
    ,updater="grow_colmaker"
    ,refresh_leaf=1
    ,process_type="default"
    ,grow_policy="depthwise"
    ,max_leaves=0
    ,max_bin=256
    ,predictor="cpu_predictor" #> gpu version not currently working with Julia, maybe remove completely?
    ,sample_type="uniform"
    ,normalize_type="tree"
    ,rate_drop=0.0
    ,one_drop=0
    ,skip_drop=0.0
    ,feature_selector="cyclic"
    ,top_k=0
    ,tweedie_variance_power=1.5
    ,objective="binary:logistic"
    ,base_score=0.5
    ,eval_metric="rmse"
    ,seed=0
    ,watchlist=[]
    ,num_class=1)

    model = XGBoostClassifier{Integer}(
    num_round
    ,booster
    ,silent  #> might be redundant due to verbosity
    ,disable_default_eval_metric
    ,eta
    ,gamma
    ,max_depth
    ,min_child_weight
    ,max_delta_step
    ,subsample
    ,colsample_bytree
    ,colsample_bylevel
    ,lambda
    ,alpha
    ,tree_method
    ,sketch_eps
    ,scale_pos_weight
    ,updater
    ,refresh_leaf
    ,process_type
    ,grow_policy
    ,max_leaves
    ,max_bin
    ,predictor #> gpu version not currently working with Julia, maybe remove completely?
    ,sample_type
    ,normalize_type
    ,rate_drop
    ,one_drop
    ,skip_drop
    ,feature_selector
    ,top_k
    ,tweedie_variance_power
    ,objective
    ,base_score
    ,eval_metric
    ,seed
    ,watchlist
    ,num_class)

     message = MLJ.clean!(model)           #> future proof by including these
     isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end




function MLJ.clean!(model::XGBoostRegressor)
    warning = ""
    if(model.booster=="gblinear" &&(model.updater!=("shotgun") && model.updater!=("coord_descent")))
        model.updater="shotgun"
        warning *= "updater has been changed to shotgun, the default option for booster=\"gblinear\""
    end
    if(model.objective in ["multi:softmax","multi:softprob","binary:logistic","binary:logitraw","binary:hinge"])
            warning *="\n objective function is more suited to XGBoostClassifier"
    end
    if(model.objective in ["multi:softmax","multi:softprob"])
        if model.num_class==1
            model.num_class=2
            warning *= "\n num_class has been changed to 2"
        end
        if model.eval_metric=="rmse"
            model.eval_metric="mlogloss"
            warning *= "\n eval_metric has been changed to mlogloss"
        end
    end
    return warning
end

function MLJ.clean!(model::XGBoostClassifier)
    warning = ""
    if(model.booster=="gblinear" &&(model.updater!=("shotgun") && model.updater!=("coord_descent")))
        model.updater="shotgun"
        warning *= "updater has been changed to shotgun, the default option for booster=\"gblinear\""
    end
    if(!(model.objective in ["multi:softmax","multi:softprob","binary:logistic","binary:logitraw","binary:hinge"]))
            warning *="\n objective function is more suited to XGBoostClassifier"
    end
    if(model.objective in ["multi:softmax","multi:softprob"])
        if model.num_class==1
            model.num_class=2
            warning *= "\n num_class has been changed to 2"
        end
        if model.eval_metric=="rmse"
            model.eval_metric="mlogloss"
            warning *= "\n eval_metric has been changed to mlogloss"
        end
    end
    return warning
end

#> The following optional method (the fallback does nothing, returns
#> empty warning) is called by the constructor above but also by the
#> fit methods below:

#> A required `fit` method returns `fitresult, cache, report`. (Return
#> `cache=nothing` unless you are overloading `update`)

function MLJ.fit(model::Union{XGBoostRegressor,XGBoostClassifier}
             , verbosity            #> must be here even if unsupported in pkg
             , X::Array{<:Real,2}
             , y::Vector)

    dm = XGBoost.DMatrix(X,label=y)
    fitresult = XGBoost.xgboost( dm
                               , model.num_round
                               , booster = model.booster
                               , silent = verbosity
                               , disable_default_eval_metric = model.disable_default_eval_metric
                               , eta = model.eta
                               , gamma = model.gamma
                               , max_depth = model.max_depth
                               , min_child_weight = model.min_child_weight
                               , max_delta_step = model.max_delta_step
                               , subsample = model.subsample
                               , colsample_bytree = model.colsample_bytree
                               , colsample_bylevel = model.colsample_bylevel
                               , lambda = model.lambda
                               , alpha = model.alpha
                               , tree_method = model.tree_method
                               , sketch_eps = model.sketch_eps
                               , scale_pos_weight = model.scale_pos_weight
                               , updater = model.updater
                               , refresh_leaf = model.refresh_leaf
                               , process_type = model.process_type
                               , grow_policy = model.grow_policy
                               , max_leaves = model.max_leaves
                               , max_bin = model.max_bin
                               , predictor = model.predictor
                               , sample_type = model.sample_type
                               , normalize_type = model.normalize_type
                               , rate_drop = model.rate_drop
                               , one_drop = model.one_drop
                               , skip_drop = model.skip_drop
                               , feature_selector = model.feature_selector
                               , top_k = model.top_k
                               , tweedie_variance_power = model.tweedie_variance_power
                               , objective = model.objective
                               , base_score = model.base_score
                               , eval_metric=model.eval_metric
                               , seed = model.seed
                               , watchlist=model.watchlist
                               , num_class=model.num_class)

    #> return package-specific statistics (eg, feature rankings,
    #> internal estimates of generalization error) in `report`, which
    #> should be `nothing` or a dictionary keyed on symbols.

    cache = nothing
    report = nothing

    return fitresult, cache, report

end

function MLJ.fit(model::Union{XGBoostRegressor,XGBoostClassifier}
             , verbosity            #> must be here even if unsupported in pkg
             , dm::XGBoost.DMatrix)

    fitresult = XGBoost.xgboost( dm
                               , model.num_round
                               , booster = model.booster
                               , silent = verbosity
                               , disable_default_eval_metric = model.disable_default_eval_metric
                               , eta = model.eta
                               , gamma = model.gamma
                               , max_depth = model.max_depth
                               , min_child_weight = model.min_child_weight
                               , max_delta_step = model.max_delta_step
                               , subsample = model.subsample
                               , colsample_bytree = model.colsample_bytree
                               , colsample_bylevel = model.colsample_bylevel
                               , lambda = model.lambda
                               , alpha = model.alpha
                               , tree_method = model.tree_method
                               , sketch_eps = model.sketch_eps
                               , scale_pos_weight = model.scale_pos_weight
                               , updater = model.updater
                               , refresh_leaf = model.refresh_leaf
                               , process_type = model.process_type
                               , grow_policy = model.grow_policy
                               , max_leaves = model.max_leaves
                               , max_bin = model.max_bin
                               , predictor = model.predictor
                               , sample_type = model.sample_type
                               , normalize_type = model.normalize_type
                               , rate_drop = model.rate_drop
                               , one_drop = model.one_drop
                               , skip_drop = model.skip_drop
                               , feature_selector = model.feature_selector
                               , top_k = model.top_k
                               , tweedie_variance_power = model.tweedie_variance_power
                               , objective = model.objective
                               , base_score = model.base_score
                               , eval_metric=model.eval_metric
                               , seed = model.seed
                               , watchlist=model.watchlist
                               , num_class = model.num_class)

    #> return package-specific statistics (eg, feature rankings,
    #> internal estimates of generalization error) in `report`, which
    #> should be `nothing` or a dictionary keyed on symbols.

    cache = nothing
    report = nothing

    return fitresult, cache, report

end



#Order of predict matters, see https://discourse.julialang.org/t/fun-with-kwargs-methods/15711/5

function MLJ.predict(model::Union{XGBoostRegressor,XGBoostClassifier}
        , fitresult::XGBoost.Booster
        , Xnew::Union{XGBoost.DMatrix,Array{<:Real,2}}
        ; ntree_limit::Integer)

    return XGBoost.predict(fitresult, Xnew,ntree_limit=ntree_limit)
end


function MLJ.predict(model::Union{XGBoostRegressor,XGBoostClassifier}
        , fitresult::XGBoost.Booster
        , Xnew::Union{XGBoost.DMatrix,Array{<:Real,2}})

    return XGBoost.predict(fitresult, Xnew)
end

#module end
end

using .XGBoost_
export XGBoostClassifier,XGBoostRegressor
