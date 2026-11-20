# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

"""
    struct HeterogeneousComputing.NoPrecision{T}

Indicates that no specific numerical precision associated with an object of
type `T` could be determined.
"""
struct NoPrecision{T} end

Base.promote_rule(::Type{<:NoPrecision}, ::Type{T}) where {T<:AbstractFloat} = T
Base.typejoin(::Type{NoPrecision{T}}, ::Type{<:NoPrecision}) where T = NoPrecision{T}


"""
    get_precision(x::T)::Union{
        Type{<:AbstractFloat},
        Type{<:HeterogeneousComputing.NoPrecision}
    }

Returns the numerical precision used by `x` or [`NoPrecision{T}`](@ref)
if no numerical precision can be determined for `x`.

In general, do not specialize `get_precision`, specialize
[`get_precision_fromtype`](@ref) instead.
"""
function get_precision end
export get_precision

get_precision(::T) where {T} = get_precision_fromtype(T)
get_precision(::Type{T}) where T = get_precision_fromtype(Type{T})


"""
    get_precision_fromtype(::Type{T})

Returns the numerical precision associated with type `T` or
[`NoPrecision{T}`](@ref) if no numerical precision can be determined for type
`T`.

The generic rule combines the field precisions of concrete types and returns
`NoPrecision{T}` for abstract types. Array types use their element precision.

```jldoctest
julia> get_precision_fromtype(AbstractVector{Float32})
Float32
```
"""
function get_precision_fromtype end
export get_precision_fromtype

# ToDo: Improve type stability of generic implementation.
function get_precision_fromtype(::Type{T}) where T
    isconcretetype(T) || return NoPrecision{T}
    _precision_needs_guard(T) || return _get_precision_from_fieldtypes(T, fieldtypes(T))
    path = get(task_local_storage(), _precision_needs_guard, ())
    T in path && return NoPrecision{T}
    return task_local_storage(_precision_needs_guard, (path..., T)) do
        return _get_precision_from_fieldtypes(T, fieldtypes(T))
    end
end

# The generated check uses the declared array element type.
_array_precision_parameter(::Type{<:AbstractArray}) = nothing
_array_precision_parameter(::Type{<:AbstractArray{T}}) where T = T

# Keep small acyclic field walks inferable without consulting task-local state.
@generated function _precision_needs_guard(::Type{T}) where T
    seen = Base.IdSet{Any}()
    pending = Any[fieldtypes(T)...]
    while !isempty(pending)
        FT = pop!(pending)
        FT === T && return true
        FT in seen && continue
        # Limit the static search for large or recursively parameterized field graphs.
        length(seen) >= 32 && return true
        push!(seen, FT)
        if FT isa DataType
            for parameter in FT.parameters
                parameter isa Type && push!(pending, parameter)
            end
        end
        if FT !== Union{} && FT <: AbstractArray
            ET = _array_precision_parameter(FT)
            isnothing(ET) && return true
            push!(pending, ET)
        elseif isconcretetype(FT) && !isbitstype(FT)
            append!(pending, fieldtypes(FT))
        end
    end
    return false
end

_get_precision_from_fieldtypes(::Type{T}, ::Tuple{}) where T = NoPrecision{T}

function _get_precision_from_fieldtypes(::Type{T}, ftypes::Tuple) where T
    return promote_type(map(get_precision_fromtype, ftypes)...)
end

get_precision_fromtype(::Type{T}) where {T<:AbstractFloat} = isconcretetype(T) ? T : NoPrecision{T}
get_precision_fromtype(::Type{Tuple{}}) = NoPrecision{Tuple{}}
get_precision_fromtype(::Type{T}) where {T<:Integer} = NoPrecision{T}
get_precision_fromtype(::Type{T}) where {T<:Union{AbstractString,Symbol}} = NoPrecision{T}

get_precision_fromtype(::Type{Complex{T}}) where T = get_precision_fromtype(T)
get_precision_fromtype(AT::Type{<:AbstractArray}) = get_precision_fromtype(eltype(AT))
