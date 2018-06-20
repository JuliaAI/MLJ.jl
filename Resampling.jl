"""
    returns lists of train and test arrays, based on the sampling method
"""
function get_samples(sampler::Resampling, n_obs::Int64)
    trainᵢ = []
    testᵢ = []
    if sampler.method == "KFold"
        kfold = Kfold(n_obs, sampler.iterations)
        for train in kfold
            push!(trainᵢ, collect(train))
            push!(testᵢ, setdiff(1:n_obs, trainᵢ[end]))
        end
    elseif sample.method == "LOOCV"
        loocv = LOOCV(n_obs)
        for train in loocv
            push!(trainᵢ, collect(train))
            push!(testᵢ, setdiff(1:n_obs, trainᵢ[end]))
        end
    end
    trainᵢ, testᵢ
end
