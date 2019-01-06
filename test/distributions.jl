module TestDistributions

# using Revise
using Test
using MLJ
using StatsBase

v = collect("asdfghjklzxc")
d = UnivariateNominal(v, [0.09, 0.02, 0.1, 0.1, 0.1, 0.1, 0.1, 0.11, 0.01, 0.1, 0.07, 0.1])
@test pdf(d, 's') ≈ 0.02
@test mode(d) == 'k'
rand(d, 5)

v = collect("abcd")
d = UnivariateNominal(v, [0.2, 0.3, 0.1, 0.4])
sample = rand(d, 10^4)
freq_given_label = countmap(sample)
# if this fails it is bug or an exceedingly rare event or a bug:
@test MLJ.keys_ordered_by_values(freq_given_label) == ['c', 'a', 'b', 'd']
                                                 

end # module

true
