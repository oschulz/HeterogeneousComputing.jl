# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).


"""
    struct HeterogeneousComputing.NoRNG{T}

Indicates that no specific random number generator associated with an object
of type `T` could be determined.

See [`get_rng`](@ref) for details.
"""
struct NoRNG{T} end
NoRNG(::T) where T = NoRNG{T}()
NoRNG(::Type{T}) where T = NoRNG{Type{T}}()


"""
    get_rng(x::T)

Tries to determine the random number generator used by `x`.

Returns [`NoRNG{T}()`](@ref) if `x` doesn't seem to be associated with a
specific random number generator.
"""
function get_rng end
export get_rng

# Do *not* do a DFS search through structs, using internal RNGs of objects.
# Using internal RNGs outside of them could lead to undesired behavior
# resp. side-effects, e.g. if objects use counter-based RNGs in specific
# ways.
@inline get_rng(::T) where T = NoRNG{T}()
@inline get_rng(rng::AbstractRNG) = rng
