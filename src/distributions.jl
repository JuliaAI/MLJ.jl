## UNIVARIATE NOMINAL PROBABILITY DISTRIBUTION

"""
    UnivariateNominal(prob_given_label)

A discrete univariate distribution whose finite support is the set of keys of the
provided dictionary, `prob_given_label`. The dictionary values specify
the corresponding probabilities, which must be nonnegative and sum to
one.

    UnivariateNominal(labels, p)

A discrete univariate distribution whose finite support is the
elements of the vector `labels`, and whose corresponding probabilities
are elements of the vector `p`.

````julia
d = UnivariateNominal(["yes", "no", "maybe"], [0.1, 0.2, 0.7])
pdf(d, "no") # 0.2
mode(d) # "maybe"
rand(d, 5) # ["maybe", "no", "maybe", "maybe", "no"]
````

"""
struct UnivariateNominal{L,T<:Real} <: MLJ.MLJType
    prob_given_label::Dict{L,T}
end

function UnivariateNominal(labels::Vector{L}, p::Vector{T}) where {L,T<:Real}
        Distributions.@check_args(UnivariateNominal, Distributions.isprobvec(p))
        Distributions.@check_args(UnivariateNominal, length(labels)==length(p))
        prob_given_label = Dict{L,T}()
        for i in eachindex(p)
            prob_given_label[labels[i]] = p[i]
        end
        return  UnivariateNominal(prob_given_label)
end

function Distributions.mode(d::UnivariateNominal)
    dic = d.prob_given_label
    p = values(dic)
    max_prob = maximum(p)
    m = first(first(dic)) # mode, just some label for now
    for (x, prob) in dic
        if prob == max_prob
            m = x
            break
        end
    end
    return m
end

Distributions.pdf(d::UnivariateNominal, x) = d.prob_given_label[x]

"""
    _cummulative(d::UnivariateNominal)

Return the cummulative probability vector `[0, ..., 1]` for the
distribution `d`, using whatever ordering is used in the dictionary
`d.prob_given_label`. Used only for to implement random sampling from
`d`.

"""
function _cummulative(d::UnivariateNominal{L,T}) where {L,T<:Real}
    p = collect(values(d.prob_given_label))
    K = length(p)
    p_cummulative = Array{T}(undef, K + 1)
    p_cummulative[1] = zero(T)
    p_cummulative[K + 1] = one(T)
    for i in 2:K
        p_cummulative[i] = p_cummulative[i-1] + p[i-1]
    end
    return p_cummulative
end


"""
_rand(p_cummulative)

Randomly sample the distribution with discrete support `1:n` which has
cummulative probability vector `p_cummulative=[0, ..., 1]` (of length
`n+1`). Does not check the first and last elements of `p_cummulative`
but does not use them either. 

"""
function _rand(p_cummulative)
    real_sample = rand()
    K = length(p_cummulative)
    index = K
    for i in 2:K
        if real_sample < p_cummulative[i]
            index = i - 1
            break
        end
    end
    return index
end

function Base.rand(d::UnivariateNominal)
    p_cummulative = _cummulative(d)
    labels = collect(keys(d.prob_given_label))
    return labels[_rand(p_cummulative)]
end

function Base.rand(d::UnivariateNominal, n::Int)
    p_cummulative = _cummulative(d)
    labels = collect(keys(d.prob_given_label))
    return [labels[_rand(p_cummulative)] for i in 1:n]
end
    
        
    
    
