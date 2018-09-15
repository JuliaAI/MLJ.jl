function Fakedata(N,d)
    n_obs = 100
    x = randn((n_obs,d))
    y = sum(x*randn(d),2)+randn(n_obs)*0.1

    hcat(x,y)
end

function FakedataClassif(N,d)
    n_obs = 100
    x = randn((n_obs,d))
    y = ( sum(x*randn(d),2) .> mean(sum(x*randn(d),2)) )

    hcat(x,y)
end

function load_interface_for{T<:BaseModel}(model::T)
    if isa(model, DecisionTreeModel)
        print("Including library for $(typeof(model)) \n")
        include("src/interfaces/decisiontree_interface.jl")
    elseif isa(model, SparseRegressionModel)
        print("Including library for $(typeof(model)) \n")
        include("src/interfaces/glm_interface.jl")
    end
end

function load_interface_for(model::String)
    if model == "SparseRegressionModel"
        print("Including library for "*model*"\n")
        include("src/interfaces/glm_interface.jl")
    end
end