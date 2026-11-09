# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).


"""
    has_rng_streamkey(rng::AbstractRNG)::Bool

Whether `rng` carries a stream key.

A keyed RNG identifies its random stream by a 256-bit key that drawing numbers
never changes, so [`subrng`](@ref) can derive child streams from it
reproducibly. Ordinary RNGs carry no key, [`rootrng`](@ref) creates keyed
ones.

See also [`rng_streamkey`](@ref).
"""
function has_rng_streamkey end
export has_rng_streamkey

has_rng_streamkey(::AbstractRNG) = false


"""
    rng_streamkey(rng::AbstractRNG)::NTuple{4,UInt64}

The stream key of a keyed RNG, see [`has_rng_streamkey`](@ref).

The key identifies the random stream of `rng`: [`rootrng`](@ref) creates the
same stream from the same key, and the key of an RNG derived with
[`subrng`](@ref) depends only on the key of its root and on the ids along the
way. Throws an `ArgumentError` for RNGs without a key.
"""
function rng_streamkey end
export rng_streamkey

function rng_streamkey(rng::AbstractRNG)
    return throw(ArgumentError("$(nameof(typeof(rng))) has no stream key, see rootrng"))
end


"""
    subrng(rng::AbstractRNG, id, ids...)

Derive a child RNG from the keyed RNG `rng`, identified by `id` and further
`ids`.

Ids are integers in `0:typemax(UInt64)` or symbols. The stream key of the
child depends only on the key of `rng` and the ids, not on the numbers drawn
from `rng` before, and deriving a child doesn't advance `rng`. Several ids
derive step by step, so `subrng(rng, :walker, 3)` equals
`subrng(subrng(rng, :walker), 3)`. The same ids always yield the same child,
so an algorithm that runs repeatedly has to include something like a step
number in the ids if its runs are to differ.

The child uses the same kind of random number generator as `rng`, with a
fresh state derived from its key. Throws an `ArgumentError` if `rng` has no
stream key, see [`rootrng`](@ref).
"""
function subrng end
export subrng

function subrng(rng::AbstractRNG, @nospecialize(id), @nospecialize(ids...))
    return throw(ArgumentError("$(nameof(typeof(rng))) has no stream key to derive child RNGs from, see rootrng"))
end


"""
    rootrng([cunit::AbstractComputeUnit,] seed::Integer)
    rootrng([cunit::AbstractComputeUnit,] entropy::AbstractRNG)
    rootrng([cunit::AbstractComputeUnit,] key::NTuple{4,UInt64})
    rootrng([cunit::AbstractComputeUnit])

Create a keyed RNG on compute unit `cunit`, the root of a tree of RNGs
derived with [`subrng`](@ref).

The stream key of the root is derived from the integer `seed` (in
`0:typemax(UInt64)`), drawn from the RNG `entropy`, which advances, or drawn
from the default RNG of `cunit` if neither is given. `cunit` defaults to
`CPUnit()`, or to the compute unit of `entropy`.

A root created from a key, typically the [`rng_streamkey`](@ref) of another
keyed RNG, restarts the stream that `rootrng` or [`subrng`](@ref) created for
that key on `cunit`. It doesn't resume a stream that has advanced, and it
doesn't recreate the generator of an RNG that `adapt` moved to another
compute unit or that `Random.seed!` reseeded, since their state no longer
follows from the key.

The root draws with the kind of RNG that `cunit` uses for derived generators,
in a state derived from the key. See [`KeyedRNG`](@ref).
"""
function rootrng end
export rootrng


"""
    struct KeyedRNG{R<:AbstractRNG} <: AbstractRNG

A random number generator with a stream key, see [`rng_streamkey`](@ref).

Wraps an RNG that generates the numbers and forwards the `Random` API to it.
`Random.seed!` reseeds the wrapped RNG, the key stays. Keyed RNGs are created
with [`rootrng`](@ref) and [`subrng`](@ref).
"""
struct KeyedRNG{R<:AbstractRNG} <: AbstractRNG
    rng::R
    streamkey::NTuple{4,UInt64}
end
export KeyedRNG

has_rng_streamkey(::KeyedRNG) = true
rng_streamkey(rng::KeyedRNG) = rng.streamkey

# Backends that compile code (Reactant) can't trace keyed RNGs. Their
# extensions make these hooks throw within compiled code:
struct _KeyedDraw end
struct _KeyedCreate end
_assert_not_compiling(@nospecialize(x)) = nothing

# The RNG that draws for a keyed RNG. Extensions reject wrapped RNGs that
# can't draw where they are used:
_draw_rng(rng::KeyedRNG) = rng.rng

# Moving a keyed RNG moves the wrapped RNG, the key stays:
Adapt.adapt_structure(to, rng::KeyedRNG) = KeyedRNG(adapt(to, rng.rng), rng.streamkey)

Base.copy(rng::KeyedRNG) = KeyedRNG(copy(rng.rng), rng.streamkey)
Base.:(==)(a::KeyedRNG, b::KeyedRNG) = a.streamkey == b.streamkey && a.rng == b.rng
Base.hash(rng::KeyedRNG, h::UInt) = hash(rng.rng, hash(rng.streamkey, hash(:KeyedRNG, h)))

function Base.show(io::IO, rng::KeyedRNG)
    return print(io, "KeyedRNG(", rng.rng, ", streamkey = ", rng.streamkey, ")")
end

# Fills of dense host arrays (which Xoshiro fills in bulk) and of GPU arrays
# go to the wrapped RNG as a whole, so its own methods and kernels are used.
# Random fills other arrays element by element, through the scalar draws
# forwarded below. Forwarding fills of all arrays would be ambiguous with
# packages that fill their own array types from any RNG.
@static if isdefined(Base, :Memory)
    const _DenseHostArray = Union{
        Array,Memory,SubArray{<:Any,<:Any,<:Union{Array,Memory},<:Any,true},Random.UnsafeView
    }
else
    const _DenseHostArray = Union{Array,SubArray{<:Any,<:Any,<:Array,<:Any,true},Random.UnsafeView}
end

const _NativeIntegers = (Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Int128, UInt128)
const _NativeFloats = (Float16, Float32, Float64)

for f! in (:rand!, :randn!, :randexp!)
    @eval function Random.$f!(rng::KeyedRNG, A::_DenseHostArray)
        _assert_not_compiling(_KeyedDraw())
        return Random.$f!(_draw_rng(rng), A)
    end
end
function Random.rand!(rng::KeyedRNG, A::_DenseHostArray, ::Type{X}) where {X}
    _assert_not_compiling(_KeyedDraw())
    return Random.rand!(_draw_rng(rng), A, X)
end
# Fills with the samplers that Xoshiro implements in bulk:
for S in (
    (Random.SamplerType{T} for T in _NativeIntegers)...,
    (Random.SamplerTrivial{Random.CloseOpen01{T}} for T in _NativeFloats)...
)
    @eval function Random.rand!(rng::KeyedRNG, A::_DenseHostArray, sp::$S)
        _assert_not_compiling(_KeyedDraw())
        return Random.rand!(_draw_rng(rng), A, sp)
    end
end

function Random.rand!(rng::KeyedRNG, A::GPUArraysCore.AnyGPUArray)
    _assert_not_compiling(_KeyedDraw())
    return Random.rand!(_draw_rng(rng), A)
end
function Random.randn!(
    rng::KeyedRNG,
    A::GPUArraysCore.AnyGPUArray{T}
) where {T<:Union{AbstractFloat,Complex{<:AbstractFloat}}}
    _assert_not_compiling(_KeyedDraw())
    return Random.randn!(_draw_rng(rng), A)
end
function Random.randexp!(rng::KeyedRNG, A::GPUArraysCore.AnyGPUArray)
    _assert_not_compiling(_KeyedDraw())
    return Random.randexp!(_draw_rng(rng), A)
end

# Allocating draws go to the wrapped RNG as well, so GPU RNGs allocate on
# their device. Packages define such draws of their own array types for any
# RNG, so the element types are limited to ones that can't be arrays:
const _ScalarElement = Union{Number,Char,Tuple}

for f in (:rand, :randn, :randexp)
    @eval begin
        function Random.$f(rng::KeyedRNG, dims::Dims)
            _assert_not_compiling(_KeyedDraw())
            return Random.$f(_draw_rng(rng), dims)
        end
        function Random.$f(rng::KeyedRNG, d::Integer, dims::Integer...)
            _assert_not_compiling(_KeyedDraw())
            return Random.$f(_draw_rng(rng), d, dims...)
        end
        function Random.$f(rng::KeyedRNG, ::Type{X}, dims::Dims) where {X<:_ScalarElement}
            _assert_not_compiling(_KeyedDraw())
            return Random.$f(_draw_rng(rng), X, dims)
        end
        function Random.$f(rng::KeyedRNG, ::Type{X}, d::Integer, dims::Integer...) where {X<:_ScalarElement}
            _assert_not_compiling(_KeyedDraw())
            return Random.$f(_draw_rng(rng), X, d, dims...)
        end
    end
end

# Scalar draws forward the samplers that Xoshiro implements natively, Random's
# generic algorithms build the rest on top of them:
Random.rng_native_52(rng::KeyedRNG) = Random.rng_native_52(rng.rng)
Random.seed!(rng::KeyedRNG) = (Random.seed!(rng.rng); rng)
Random.seed!(rng::KeyedRNG, seed) = (Random.seed!(rng.rng, seed); rng)

for S in (
    (Random.SamplerType{T} for T in _NativeIntegers)...,
    (Random.SamplerTrivial{Random.CloseOpen01{T}} for T in _NativeFloats)...,
    Random.SamplerTrivial{Random.UInt52Raw{UInt64}}, Random.SamplerTrivial{Random.UInt52{UInt64}},
    Random.SamplerTrivial{Random.UInt104{UInt128}}
)
    @eval function Random.rand(rng::KeyedRNG, sp::$S)
        _assert_not_compiling(_KeyedDraw())
        return Random.rand(_draw_rng(rng), sp)
    end
end

get_compute_unit_impl(rng::KeyedRNG) = get_compute_unit(rng.rng)

function _fill_random!(f!::F, rng::KeyedRNG, A::AbstractArray) where {F}
    _assert_not_compiling(_KeyedDraw())
    return _fill_random!(f!, _draw_rng(rng), A)
end
function _draw_scalar(f!::F, rng::KeyedRNG, ::Type{T}, cunit::AbstractComputeUnit) where {F,T}
    _assert_not_compiling(_KeyedDraw())
    return _draw_scalar(f!, _draw_rng(rng), T, cunit)
end
_draw_seeds(rng::KeyedRNG, home::AbstractComputeUnit) = _draw_seeds(rng.rng, home)

# An RNG derived to replace a keyed one keeps its key:
_keep_streamkey(::AbstractRNG, rng::AbstractRNG) = rng
_keep_streamkey(keyed::KeyedRNG, rng::AbstractRNG) = KeyedRNG(rng, keyed.streamkey)

# Compiled code works with the wrapped RNGs. A key frozen into a compiled
# function would come back with results computed for other keys. Keys are
# stripped through Adapt, so everything that can be moved to a compute unit
# can be stripped:
struct _StripStreamKeys end
_strip_streamkeys(x) = adapt(_StripStreamKeys(), x)
Adapt.adapt_structure(::_StripStreamKeys, rng::KeyedRNG) = rng.rng
function Adapt.adapt_structure(::_StripStreamKeys, ctx::GenContext{T}) where {T}
    return GenContext{T}(ctx.cunit, _strip_streamkeys(ctx.rng))
end


# Key derivation. Threefry-4x64 serves as a pseudorandom function from a key
# and a 256-bit counter block to a new key. The first counter word names the
# domain of the remaining three, so integer ids, symbol ids, generator states
# and seeds can't collide with each other. For a fixed parent key, distinct
# counter blocks give distinct keys:
const _STREAMKEY_INT_ID = UInt64(1)
const _STREAMKEY_SYMBOL_ID = UInt64(2)
const _STREAMKEY_LEAF = UInt64(3)
const _STREAMKEY_SEED = UInt64(4)

# Public fixed key for seed-derived roots (hex digits of pi):
const _STREAMKEY_SEED_KEY = (0x243f6a8885a308d3, 0x13198a2e03707344, 0xa4093822299f31d0, 0x082efa98ec4e6c89)

_derive_streamkey(key::NTuple{4,UInt64}, ctr::NTuple{4,UInt64}) = _threefry4x64(key, ctr)

function _child_streamkey(key::NTuple{4,UInt64}, id::Integer)
    return _derive_streamkey(key, (_STREAMKEY_INT_ID, _checked_streamid(id), UInt64(0), UInt64(0)))
end
function _child_streamkey(key::NTuple{4,UInt64}, id::Symbol)
    return _derive_streamkey(key, (_STREAMKEY_SYMBOL_ID, _symbol_digest(id)...))
end

function _child_streamkey(::NTuple{4,UInt64}, @nospecialize(id))
    return throw(ArgumentError("Stream ids must be integers or symbols, got a $(typeof(id))"))
end

_path_streamkey(key::NTuple{4,UInt64}) = key
_path_streamkey(key::NTuple{4,UInt64}, id, ids...) = _path_streamkey(_child_streamkey(key, id), ids...)

function _checked_streamid(id::Integer)
    id isa Bool && throw(ArgumentError("Stream ids must be integers or symbols, got a Bool"))
    0 <= id <= typemax(UInt64) || throw(ArgumentError("Integer stream ids must be in 0:typemax(UInt64), got $id"))
    return UInt64(id)
end

# The first 192 bits of the SHA-256 digest of the symbol's name, as three
# little-endian words:
function _symbol_digest(sym::Symbol)
    digest = sha256(string(sym))
    return ntuple(Val(3)) do i
        word = UInt64(0)
        for j in 8:-1:1
            word = word << 8 | digest[8 * (i - 1) + j]
        end
        return word
    end
end

function _seed_streamkey(seed::Integer)
    return _derive_streamkey(_STREAMKEY_SEED_KEY, (_STREAMKEY_SEED, _checked_streamid(seed), UInt64(0), UInt64(0)))
end

# The state of the wrapped RNG of a keyed RNG with this key:
_leaf_seeds(key::NTuple{4,UInt64}) = _derive_streamkey(key, (_STREAMKEY_LEAF, UInt64(0), UInt64(0), UInt64(0)))


function subrng(rng::KeyedRNG, id, ids...)
    _assert_not_compiling(_KeyedCreate())
    key = _path_streamkey(rng.streamkey, id, ids...)
    home = _rng_home(get_compute_unit(rng.rng), CPUnit())
    leaf = _within_unit(() -> _seeded_rng_like(rng.rng, home, _leaf_seeds(key)), home)
    return KeyedRNG(leaf, key)
end

function rootrng(cunit::AbstractComputeUnit, key::NTuple{4,UInt64})
    _assert_not_compiling(_KeyedCreate())
    return KeyedRNG(_seeded_rng(cunit, _leaf_seeds(key)), key)
end
rootrng(cunit::AbstractComputeUnit, seed::Integer) = rootrng(cunit, _seed_streamkey(seed))
function rootrng(cunit::AbstractComputeUnit, entropy::AbstractRNG)
    return rootrng(cunit, _draw_seeds(entropy, _rng_home(get_compute_unit(entropy), cunit)))
end
rootrng(cunit::AbstractComputeUnit) = rootrng(cunit, default_device_rng(cunit))
rootrng(dev::AbstractDevice, args...) = rootrng(DeviceUnit(dev), args...)

rootrng() = rootrng(CPUnit())
rootrng(seed::Integer) = rootrng(CPUnit(), seed)
rootrng(key::NTuple{4,UInt64}) = rootrng(CPUnit(), key)
rootrng(entropy::AbstractRNG) = rootrng(_rng_gen_cunit(get_compute_unit(entropy)), entropy)


"""
    rootcontext([cunit::AbstractComputeUnit,] args...)

A [`GenContext`](@ref) on `cunit` with a keyed RNG created by
`rootrng(cunit, args...)`, see [`rootrng`](@ref).

The precision is `Float64`, use `GenContext{T}(cunit, rootrng(cunit, args...))`
for other precisions.
"""
function rootcontext end
export rootcontext

rootcontext(cunit::AbstractComputeUnit, args...) = GenContext(cunit, rootrng(cunit, args...))
rootcontext(dev::AbstractDevice, args...) = rootcontext(DeviceUnit(dev), args...)
rootcontext(args...) = GenContext(rootrng(args...))


"""
    subcontext(ctx::GenContext, id, ids...)

A copy of `ctx` with the child RNG `subrng(get_rng(ctx), id, ids...)`, see
[`subrng`](@ref).
"""
function subcontext end
export subcontext

subcontext(ctx::GenContext{T}, id, ids...) where T = GenContext{T}(ctx.cunit, subrng(ctx.rng, id, ids...))
