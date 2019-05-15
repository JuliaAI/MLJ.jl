module TestSimpleRidgeRegressor

using Test
using MLJ

# These are calculated values, the coefficients (parmeters) should be [1, -1 , 2]
ip = [1 0 3 4 5; 2 1 3 -3 4; 0 1 -3 2 1]
op = [-1, 1, -6, 11, 3]

ip = MLJ.table(ip')

model = MLJ.SimpleRidgeRegressor(lambda = 0.0)

fitresult, report ,cache = MLJ.fit(model, 0, ip, op);

@test MLJ.predict(model, fitresult, Float64[-1,0,2]')[1] ≈ 3

end
true