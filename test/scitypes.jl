module TestScitypes

using MLJ
using Test

S = scitype(ConstantClassifier())
@test S().prediction_type == :probabilistic
U = scitype(FeatureSelector())
@test U().input_scitype == MLJ.Table

@test scitype(OneHotEncoder()) ==
    MLJ.UnsupervisedScitype{Table,Table}

@test scitype(ConstantRegressor()) ==
    MLJ.SupervisedScitype{Table,
                          AbstractVector{Continuous},
                          :probabilistic}

end
true
