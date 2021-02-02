module TestScitypes

using MLJ
import MLJBase
using Test

S = scitype(ConstantClassifier())
@test S().prediction_type == :probabilistic
U = scitype(FeatureSelector())
@test U().input_scitype == MLJ.Table

# XXX fix this in the next release cycle (Nov 5, 2019)
# M = scitype(rms)
# @test_broken M().prediction_type == :deterministic

@test scitype(OneHotEncoder()) ==
    MLJ.UnsupervisedScitype{Table,Table}

@test scitype(ConstantRegressor()) ==
    MLJ.SupervisedScitype{Table,
                          AbstractVector{Continuous},
                          :probabilistic}

end
true
