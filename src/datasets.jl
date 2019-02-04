datadir = joinpath(srcdir, "../data/") # TODO: make OS agnostic

"""Load a well-known public regression dataset with nominal features."""
function load_boston()
    df = CSV.read(joinpath(datadir, "Boston.csv"),
                  categorical=false, allowmissing=:none)
    return SupervisedTask(data=df, target=:MedV, ignore=[:Chas], properties=())
end

"""Load a reduced version of the well-known Ames Housing task,
having six numerical and six categorical features."""
function load_ames()
    df = CSV.read(joinpath(datadir, "reduced_ames.csv"),
                  categorical=false, allowmissing=:none)
    df[:target] = exp.(df[:target])
    return SupervisedTask(data=df, target=:target, properties=())
end

"""Load a well-known public classification task with nominal features."""
function load_iris()
    df = CSV.read(joinpath(datadir, "iris.csv"),
                  categorical=true, allowmissing=:none)
    # TODO: fix things so that next line can be deleted
#    df[:target] = [df[:target]...] # change CategoricalArray to Array, keeping categ eltype
    return SupervisedTask(data=df, target=:target, properties=())
end

"""Load a well-known crab classification dataset with nominal features."""
function load_crabs()
    df = CSV.read(joinpath(datadir, "crabs.csv"),
                  categorical=true, allowmissing=:none)
    return SupervisedTask(data=df, target=:sp, ignore=[:sex, :index], properties=())
end

"""Get some supervised data now!!"""
function datanow()
    Xtable, y = X_and_y(load_boston())
    X = DataFrame(Xtable)  # force table to be dataframe;
                                     # should become redundant

    return (X[1:75,:], y[1:75])
end
