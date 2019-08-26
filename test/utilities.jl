module TestUtilities

# using Revise
using Test
using MLJ
using ScientificTypes
using Tables
using OrderedCollections

d = Dict(
    :x => Set([1, 2]),
    :y => Set([2, 3, 5]),
    :z => Set([4, 7]),
    :a => Set([8, 1]),
    :b => Set([4,]),
    :w => Set([3, 1, 2]),
    :t => Set([0,]))

dinv = Dict(
    0 => Set([:t,]),
    1 => Set([:x, :a, :w]),
    2 => Set([:x, :y, :w]),
    3 => Set([:y, :w]),
    4 => Set([:z, :b]),
    5 => Set([:y,]),
    7 => Set([:z,]),
    8 => Set([:a,]))

@test MLJ.inverse(d) == dinv

d = LittleDict(
    :x => Set([1, 2]),
    :y => Set([2, 3, 5]),
    :z => Set([4, 7]),
    :a => Set([8, 1]),
    :b => Set([4,]),
    :w => Set([3, 1, 2]),
    :t => Set([0,]))

dinv = LittleDict(
    0 => Set([:t,]),
    1 => Set([:x, :a, :w]),
    2 => Set([:x, :y, :w]),
    3 => Set([:y, :w]),
    4 => Set([:z, :b]),
    5 => Set([:y,]),
    7 => Set([:z,]),
    8 => Set([:a,]))

@test MLJ.inverse(d) == dinv

d = LittleDict()
d[:test] = Tuple{Union{Continuous,Missing}, Finite}
d["junk"] = LittleDict{Any,Any}("H" => Missing, :c => Table(Finite,Continuous),
                                :cross => "lemon", :t => :w, "r" => "r")
d["a"] = "b"
d[:f] = true
d["j"] = :post


@test MLJ.decode_dic(MLJ.encode_dic(d)) == d

mutable struct M
    a1
    a2
end
mutable struct A1
    a11
    a12
end
mutable struct A2
    a21
end
mutable struct A21
    a211
    a212
end

@testset "recursive getproperty, setproperty!" begin

    m = (a1 = (a11 = 10, a12 = 20), a2 = (a21 = (a211 = 30, a212 = 40),)) 

    @test MLJ.recursive_getproperty(m, :(a1.a12)) == 20
    @test MLJ.recursive_getproperty(m, :a1) == (a11 = 10, a12 = 20)
    @test MLJ.recursive_getproperty(m, :(a2.a21.a212)) == 40

    m = M(A1(10, 20), A2(A21(30, 40)))
    MLJ.recursive_setproperty!(m, :(a2.a21.a212), 42)
    @test MLJ.recursive_getproperty(m, :(a1.a11)) == 10
    @test MLJ.recursive_getproperty(m, :(a1.a12)) == 20
    @test MLJ.recursive_getproperty(m, :(a2.a21.a211)) == 30
    @test MLJ.recursive_getproperty(m, :(a2.a21.a212)) == 42
    @test MLJ.recursive_getproperty(
        MLJ.recursive_getproperty(m, :(a2.a21)), :a212) == 42

end


end # module
true
