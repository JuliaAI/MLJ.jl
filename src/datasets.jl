datadir = joinpath(srcdir, "../data/") # TODO: make OS agnostic

"""Load a well-known public regression dataset with nominal features."""
function load_boston()
    df = CSV.read(joinpath(datadir, "Boston.csv"),
                  categorical=false, allowmissing=:none)
    return SupervisedTask(data=df, target=:MedV, ignore=[:Chas], properties=[Numeric(), Regression()]) 
end

"""Load a reduced version of the well-known Ames Housing task,
having six numerical and six categorical features."""
function load_ames()
    df = CSV.read(joinpath(datadir, "reduced_ames.csv"),
                  categorical=false, allowmissing=:none)
    df[:target] = exp.(df[:target])
    return SupervisedTask(data=df, target=:target, properties=[Numeric(), Nominal(), Regression()]) 
end

"""Load a well-known public classification task with nominal features."""
function load_iris()
    df = CSV.read(joinpath(datadir, "iris.csv"),
                  categorical=true, allowmissing=:none)
    df[:target] = [df[:target]...] # change CategoricalArray to Array, keeping categ eltype
    return SupervisedTask(data=df, target=:target, properties=[Numeric(), MultiClass()])
end

"""Get some supervised data now!!"""
datanow() =  X_and_y(load_boston())


