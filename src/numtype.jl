# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

"""
    real_numtype(T::Type)

Return the underlying numerical type of T that's a subtype of `Real`.

Uses type promotion among underlying `Real` type in `T`.

Non numerical types that are commonly used to express default and missing
values or named choices/options are treated as `Bool`.

In contract to [`get_precision_fromtype`](@ref), the function `real_numtype`
may also return subtypes of `Integer` and will preserve types like
`ForwardDiff.Dual`.

Example:

```julia

A = fill(fill(rand(Float32, 5), 10), 5)
real_numtype(typeof(A)) == Float32
```
"""
function real_numtype end
export real_numtype

_no_numtype_for(Base.@nospecialize(T::Type)) = throw(ArgumentError("Can't derive underlying numeric type for type $T"))

@generated function real_numtype(::Type{T}) where {T}
    if isempty(T.parameters)
        :(_no_numtype_for($T))
    else
        :(promote_type(map(real_numtype, $((T.parameters...,)))...))
    end
end

real_numtype(::Type{T}) where {T<:Real} = T
real_numtype(::Type{<:Complex{T}}) where {T<:Real} = T
real_numtype(AT::Type{<:AbstractArray}) = real_numtype(eltype(AT))

real_numtype(::Type{<:NamedTuple{names,T}}) where {names,T} = real_numtype(T)

real_numtype(::Type{Tuple{}}) = Bool
real_numtype(::Type{Nothing}) = Bool
real_numtype(::Type{Missing}) = Bool
real_numtype(::Type{Symbol}) = Bool
real_numtype(::Type{<:Enum}) = Bool
real_numtype(::Type{<:AbstractString}) = Bool
