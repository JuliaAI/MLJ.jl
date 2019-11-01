# Syntactic sugar for arrow syntax
# we need version ≥ 1.3 in order to make use of multiple dispatch
# over abstract types


# This allows implicit: data |> machine
(mach::AbstractMachine{<:Unsupervised})(data) = transform(mach, data)
(mach::AbstractMachine{<:Supervised})(data)   = predict(mach, data)
(mach::AbstractMachine)(data::AbstractMatrix) = data |> table |> mach

# This allows implicit: data |> Unsupervised
(m::Unsupervised)(data::AbstractNode) = data |> machine(m, data)
(m::Unsupervised)(data) = source(data) |> m
(m::Unsupervised)(data::AbstractMatrix) = data |> table |> m

# This allows implicit: data |> Supervised
(m::Supervised)(data::NTuple{2,AbstractNode}) = data[1] |> machine(m, data...)
(m::Supervised)(data::Tuple{AbstractNode,Any}) = (data[1], source(data[2], kind=:target)) |> m
(m::Supervised)(data::Tuple) = (source(data[1]), data[2]) |> m
(m::Supervised)(data::Tuple{AbstractMatrix,Any}) = (data[1] |> table, data[2]) |> m

# This allows implicit: data |> inverse_transform(node)
inverse_transform(node::Node{<:NodalMachine{<:Unsupervised}}) =
    data -> inverse_transform(node.machine, data)
