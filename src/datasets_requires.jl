# see also the macro versions in datasets.jl

using .CSV

export load_boston, load_ames, load_iris
export load_reduced_ames               
export load_crabs

datadir = joinpath(@__DIR__, "..", "data")

load_boston() = CSV.read(joinpath(datadir, "Boston.csv"), copycols=true,
                         categorical=true)

function load_reduced_ames()
    df = CSV.read(joinpath(datadir, "reduced_ames.csv"), copycols=true,
                  categorical=true)
    return coerce(df, :OverallQual => OrderedFactor,
                  :GarageCars => Count,
                  :YearBuilt => Continuous,
                  :YearRemodAdd => Continuous)
end

load_ames() = CSV.read(joinpath(datadir, "ames.csv"), copycols=true,
                  categorical=true)

load_iris() = CSV.read(joinpath(datadir, "iris.csv"), pool=true, copycols=true,
                  categorical=true)

load_crabs() = CSV.read(joinpath(datadir, "crabs.csv"), pool=true,
                  copycols=true, categorical=true)


