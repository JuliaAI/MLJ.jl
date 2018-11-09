module TestTransformer

using MLJ
using Test

# relabelling with integer transformer:
y = rand(Char, 50)
allrows = eachindex(y)
test = 3:37
to_int_hypers = ToIntTransformer()
fitresult, cache, report = MLJ.fit(to_int_hypers, 1, y)
# to_int = Trainable(to_int_hypers, y)
# fit!(to_int, allrows)
z = transform(to_int_hypers, fitresult, y[test])
@test y[test] == inverse_transform(to_int_hypers, fitresult, z)
to_int_hypers.map_unseen_to_minus_one = true
fitresult, cache, report = MLJ.fit(to_int_hypers, 1, [1, 2, 3, 4, 3])
@test report[:values] == [1, 2, 3, 4]
#to_int = Trainable(to_int_hypers, [1,2,3,4])
#fitresult, cache, report = fit!(to_int, [1,2,3,4])
@test transform(to_int_hypers, fitresult, 5) == -1
@test transform(to_int_hypers, fitresult, [5,1])[1] == -1 

# `UnivariateStandardizer`:
stand = UnivariateStandardizer()
#fit!(stand, 1:3)
fitresult, cache, report = MLJ.fit(stand, 1, [0, 2, 4])
@test round.(Int, transform(stand, fitresult, [0,4,8])) == [-1.0,1.0,3.0]
@test round.(Int, inverse_transform(stand, fitresult, [-1, 1, 3])) == [0, 4, 8] 

# `Standardizer`:
X, y = X_and_y(load_ames());
train, test = partition(eachindex(y), 0.9);

# introduce a field of type `Char`:
X[:OverallQual] = map(Char, X[:OverallQual]);

stand = Standardizer()
fitresult, cache, report = MLJ.fit(stand, 1, X)
transform(stand, fitresult, X)

stand = Standardizer(features=[:GrLivArea])
fitresult, cache, report = MLJ.fit(stand, 1, X)
@test fitresult.features[fitresult.is_transformed] == [:GrLivArea]
transform(stand, fitresult, X)

end
