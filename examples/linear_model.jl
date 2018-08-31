include("../MLJ.jl")

lm = LinModel(Dict("x0" => 0, "x1" => 1))
lm_fit = fit(lm, [],[])
predict(lm_fit,[])

nlm_fit = ModelFit(NonLinModel("haha"),2)
predict(nlm_fit,[])