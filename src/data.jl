# Some of what follows is a poor-man's stab at agnostic data
# containers. When the Queryverse columns-view interface becomes
# widely implemented, a better solution, removing specific container
# dependencies, will be possible.

struct Rows end
struct Cols end
struct Names end
struct Eltypes end

# fallback to select rows of any iterable table `X` with `X[Rows, r]`:
function Base.getindex(X::T, ::Type{Rows}, r) where T

    TableTraits.isiterabletable(X) || error("Argument is not an iterable table.")

    row_iterator = @from row in X begin
        @select row
        @collect
    end
                    
    return @from row in row_iterator[r] begin
        @select row
        @collect T
    end

end

# fallback to get the number of rows of an iterable table:
function nrows(X)

    TableTraits.isiterabletable(X) || error("Argument is not an iterable table.")

    row_iterator = @from row in X begin
        @select {}
        @collect
    end
                    
    return length(row_iterator)

end
      
#Base.getindex(df::AbstractDataFrame, ::Type{Rows}, r) = df[r,:]
Base.getindex(df::AbstractDataFrame, ::Type{Cols}, c) = df[c]
Base.getindex(df::AbstractDataFrame, ::Type{Names}) = names(df)
Base.getindex(df::AbstractDataFrame, ::Type{Eltypes}) = eltypes(df)
nrows(df::AbstractDataFrame) = size(df, 1)

#Base.getindex(df::JuliaDB.NextTable, ::Type{Rows}, r) = df[r]
#Base.getindex(df::JuliaDB.NextTable, ::Type{Cols}, c) = select(df, c)
#Base.getindex(df::JuliaDB.NextTable, ::Type{Names}) = getfields(typeof(df.columns.columns))
# nrows(df::JuliaDB.NextTable) = length(df)

Base.getindex(A::AbstractMatrix, ::Type{Rows}, r) = A[r,:]
Base.getindex(A::AbstractMatrix, ::Type{Cols}, c) = A[:,c]
Base.getindex(A::AbstractMatrix, ::Type{Names}) = 1:size(A, 2)
Base.getindex(A::AbstractMatrix{T}, ::Type{Eltypes}) where T = [T for j in 1:size(A, 2)]
nrows(A::AbstractMatrix) = size(A, 1)

Base.getindex(v::AbstractVector, ::Type{Rows}, r) = v[r]
Base.getindex(v::CategoricalArray{T,1,S} where {T,S}, ::Type{Rows}, r) = @inbounds v[r]
nrows(v::AbstractVector) = length(v)
