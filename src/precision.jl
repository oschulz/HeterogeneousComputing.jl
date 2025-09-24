# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

"""
    struct HeterogeneousComputing.NoPrecision{T}

Indicates that no specific numerical precision associated with an object of
type `T` could be determined.
"""
struct NoPrecision{T} end

Base.promote_rule(::Type{<:NoPrecision}, ::Type{T}) where {T<:AbstractFloat} = T
Base.typejoin(::Type{<:NoPrecision{T}}, ::Type{<:NoPrecision}) where T = NoPrecision{T}()


"""
    get_precision(x::T)::Union{
        Type{<:AbstractFloat},
        Type{<:HeterogeneousComputing.NoPrecision}
    }

Returns the numerical precision of used by `x` or [`NoPrecision{T}`](@ref)
if no numerical precision can be determined for `x`.

In general, do not specialize `get_precision`, specialize
[`get_precision_fromtype`](@ref)  instead.
"""
function get_precision end
export get_precision

get_precision(::T) where {T} = get_precision_fromtype(T)


"""
    get_precision_fromtype(::Type{T})

Returns the numberical precision accociate with type `T` or
[`NoPrecision{T}`](@ref) if no numerical precision can be determined for type
`T`.
"""
function get_precision_fromtype end
export get_precision_fromtype

# ToDo: Improve type stability of generic implementation.
get_precision_fromtype(::Type{T}) where T = _get_precision_from_fieldtypes(T, fieldtypes(T))

_get_precision_from_fieldtypes(::Type{T}, ::Tuple{}) where T = NoPrecision{T}

function _get_precision_from_fieldtypes(::Type{T}, ftypes::Tuple) where T
    return promote_type(map(get_precision_fromtype, ftypes)...)
end

get_precision_fromtype(::Type{T}) where {T<:AbstractFloat} = T
get_precision_fromtype(::Type{Tuple{}}) = NoPrecision{Tuple{}}
get_precision_fromtype(::Type{T}) where {T<:Integer} = NoPrecision{T}
get_precision_fromtype(::Type{T}) where {T<:Union{AbstractString,Symbol}} = NoPrecision{T}

get_precision_fromtype(::Type{Complex{T}}) where T = get_precision_fromtype(T)
get_precision_fromtype(AT::Type{<:AbstractArray}) = get_precision_fromtype(eltype(AT))
