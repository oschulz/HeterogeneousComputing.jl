# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).


"""
    struct HeterogeneousComputing.NoGenContext{T}

Indicates that no generative context could be derived from an object of type
`T`.

See [`get_gencontext`](@ref) for details.
"""
struct NoGenContext{T} end
NoGenContext(::T) where T = NoGenContext{T}()
NoGenContext(::Type{T}) where T = NoGenContext{Type{T}}()



"""
    GenContext{T=Float64}(
        cunit::AbstractComputeUnit = CPUnit(),
        rng::AbstractRNG = MLDataDevices.default_device_rng(cunit)
    )
    GenContext{T=Float64}(rng::AbstractRNG)
    GenContext{T}(dev::MLDataDevices.AbstractDevice, [rng::AbstractRNG])

Context for generative computations, with numerical precision `T`.

`GenContext(rng)` derives the compute unit from `rng` (e.g. for GPU RNGs), and
uses `CPUnit()` for host RNGs. RNGs that don't record their device, like
`GPUArrays.RNG` and rocRAND generators, are taken to be on the currently
active device.

When constructed from an `MLDataDevices` device without an explicit `T`, the
precision is `eltype(dev)` if that is a floating point type, `Float64`
otherwise.

`get_precision(ctx)`, `get_compute_unit(ctx)` and `get_rng(ctx)` return the
precision, compute unit and random number generator of the context.

`rand(ctx, dims)`, `randn(ctx, dims)` and `randexp(ctx, dims)` allocate arrays
with precision `T` on the compute unit of `ctx` and fill them using its RNG.
`rand!(ctx, A)`, `randn!(ctx, A)` and `randexp!(ctx, A)` fill `A` (which
should reside on the compute unit of `ctx`) with random numbers.

`GenContext{T}(cunit, rng)` uses the supplied `rng`. With a host RNG and a
GPU destination, the backend may generate values on the host and copy them
to the GPU. Use `GenContext{T}(cunit)` for the unit's native default RNG.

`adapt(cunit, ctx)` returns `ctx` when it already targets `cunit`. A move
to another unit keeps the RNG if it is native to the new unit. Otherwise it
creates a native generator seeded with random numbers drawn from the old one.
A stream key of the generator (see [`rng_streamkey`](@ref)) stays.
A destination construction error preserves the state of a copyable source
RNG. Opaque native generators, such as cuRAND library RNGs, can advance if
the destination fails after receiving the seed.

[`rootcontext`](@ref) turns a context into one with a keyed RNG,
[`subcontext`](@ref) derives child contexts from it.
"""
struct GenContext{T<:AbstractFloat,CU<:AbstractComputeUnit,RNG<:AbstractRNG}
    cunit::CU
    rng::RNG
end

export GenContext

@inline GenContext{T}(cunit::CU, rng::RNG) where {T,CU,RNG} = GenContext{T,CU,RNG}(cunit, rng)

@inline GenContext(args...) = GenContext{Float64}(args...)
GenContext{T}() where T = GenContext{T}(CPUnit())
GenContext{T}(cunit::AbstractComputeUnit) where T = GenContext{T}(cunit, default_device_rng(cunit))
GenContext{T}(rng::AbstractRNG) where T = GenContext{T}(_rng_gen_cunit(get_compute_unit(rng)), rng)

_rng_gen_cunit(cunit::AbstractComputeUnit) = cunit
_rng_gen_cunit(::ComputeUnitIndependent) = CPUnit()
function _rng_gen_cunit(cunit::UnknownComputeUnitOf)
    return throw(ArgumentError("Compute unit of $(typeof(cunit.x)) is unknown, specify it explicitly"))
end

GenContext{T}(dev::AbstractDevice) where T = GenContext{T}(DeviceUnit(dev))
GenContext{T}(dev::AbstractDevice, rng::AbstractRNG) where T = GenContext{T}(DeviceUnit(dev), rng)
GenContext(dev::AbstractDevice) = GenContext{_device_precision(eltype(dev))}(dev)
GenContext(dev::AbstractDevice, rng::AbstractRNG) = GenContext{_device_precision(eltype(dev))}(dev, rng)

_device_precision(::Type{T}) where {T<:AbstractFloat} = T
_device_precision(::Any) = Float64

@inline GenContext{T,CU,RNG}(ctx::GenContext) where {T,CU,RNG} = GenContext{T,CU,RNG}(ctx.cunit, ctx.rng)
@inline GenContext{T}(ctx::GenContext) where T = GenContext{T}(ctx.cunit, ctx.rng)
@inline GenContext(ctx::GenContext{T}) where T = GenContext{T}(ctx)
Base.convert(::Type{GenContext{T}}, ctx::GenContext) where T = GenContext{T}(ctx)


"""
    get_gencontext(x::T)

Get the generative context associated with `x` or [`NoGenContext{T}`](@ref)
if no context can be determined for `x`.
"""
function get_gencontext end
export get_gencontext

get_gencontext(x) = _generic_get_gencontext(x, get_precision(x), _gen_cunit(get_compute_unit(x)), get_rng(x))

# Data on multiple compute units has no generative context:
_gen_cunit(cunit) = cunit
_gen_cunit(::MixedComputeSystem) = nothing

function _generic_get_gencontext(
    @nospecialize(x),
    ::Type{T},
    cunit::AbstractComputeUnit,
    rng::AbstractRNG
) where {T<:AbstractFloat}
    return GenContext{T}(cunit, rng)
end

function _generic_get_gencontext(
    @nospecialize(x),
    ::Type{T},
    cunit::AbstractComputeUnit,
    ::NoRNG
) where {T<:AbstractFloat}
    return GenContext{T}(cunit)
end

_generic_get_gencontext(x, ::Type, ::Any, ::Any) = NoGenContext(x)

get_gencontext(ctx::GenContext) = ctx


get_precision_fromtype(::Type{<:GenContext{T}}) where T = T
get_compute_unit_impl(ctx::GenContext) = ctx.cunit
get_rng(ctx::GenContext) = ctx.rng

# Adapting a context to a compute unit or device moves it to that unit. The
# RNG stays if it is native to that unit, otherwise a native RNG is derived
# from it:
function Adapt.adapt_structure(to, ctx::GenContext{T}) where T
    cunit = _adapted_cunit(_adapt_target(to), ctx.cunit)
    cunit == ctx.cunit && return ctx
    # Some RNGs report the active device of their backend as their home.
    rng = _within_unit(ctx.cunit) do
        rng_cunit = get_compute_unit(ctx.rng)
        return _rng_native_to(rng_cunit, cunit) ? ctx.rng : _derived_rng(ctx, cunit, _rng_home(rng_cunit, ctx.cunit))
    end
    return GenContext{T}(cunit, rng)
end

_adapt_target(dev::AbstractDevice) = DeviceUnit(dev)
_adapt_target(@nospecialize(to)) = to

_adapted_cunit(to::AbstractComputeUnit, ::AbstractComputeUnit) = to
_adapted_cunit(@nospecialize(to), cunit::AbstractComputeUnit) = cunit

# Host RNGs are native to host units, device RNGs to their own unit:
_rng_native_to(::ComputeUnitIndependent, cunit::AbstractComputeUnit) = is_host_unit(cunit)
function _rng_native_to(rng_cunit::AbstractComputeUnit, cunit::AbstractComputeUnit)
    return rng_cunit == cunit || (is_host_unit(rng_cunit) && is_host_unit(cunit))
end
_rng_native_to(::Any, ::AbstractComputeUnit) = false

# A new RNG native to `cunit`, seeded with random numbers drawn from the RNG
# of `ctx`:
function _derived_rng(ctx::GenContext, cunit::AbstractComputeUnit, home::AbstractComputeUnit)
    rng = _with_seed_draw(seeds -> _seeded_rng(cunit, seeds), ctx.rng, home)
    return _keep_streamkey(ctx.rng, rng, cunit)
end

# Commit the source advance only after the destination was constructed.
_rng_snapshot(rng::AbstractRNG) = applicable(copy, rng) ? copy(rng) : nothing
function _commit_seed_draw!(rng::AbstractRNG, staged::AbstractRNG, home::AbstractComputeUnit)
    return applicable(copy!, rng, staged) ? copy!(rng, staged) : _draw_seeds(rng, home)
end

function _with_seed_draw(f::F, rng::AbstractRNG, home::AbstractComputeUnit) where F
    staged = _rng_snapshot(rng)
    if isnothing(staged) || staged === rng
        # Opaque generators cannot be rewound. Raise errors independent of
        # the seed when inference proves that no seed can return a result.
        if Base.promote_op(f, NTuple{4,UInt64}) === Union{}
            f(ntuple(_ -> UInt64(0), Val(4)))
        end
        return f(_draw_seeds(rng, home))
    end
    result = f(_draw_seeds(staged, home))
    _commit_seed_draw!(rng, staged, home)
    return result
end

# The compute unit on which an RNG generates numbers:
_rng_home(rng_cunit::AbstractComputeUnit, ::AbstractComputeUnit) = rng_cunit
_rng_home(::ComputeUnitIndependent, ::AbstractComputeUnit) = CPUnit()
_rng_home(::Any, cunit::AbstractComputeUnit) = cunit

# Draws four 64-bit seeds as 32-bit integers, which all RNGs support:
function _draw_seeds(rng::AbstractRNG, home::AbstractComputeUnit)
    A = allocate_array(home, UInt32, (8,))
    _within_unit(() -> _fill_random!(Random.rand!, rng, A), home)
    u = Array(A)
    return ntuple(i -> UInt64(u[2i - 1]) | UInt64(u[2i]) << 32, Val(4))
end

# A new RNG of the same kind as the default RNG of `cunit`:
function _seeded_rng(cunit::AbstractComputeUnit, seeds::NTuple{4,UInt64})
    rng_type = Base.promote_op(default_device_rng, typeof(cunit))
    return _within_unit(() -> _seeded_rng_like(rng_type, cunit, seeds), cunit)
end

# Use the actual default when its inferred type has no seeded constructor.
function _seeded_rng_like(::Type, cunit::AbstractComputeUnit, seeds::NTuple{4,UInt64})
    return _seeded_rng_like(default_device_rng(cunit), cunit, seeds)
end

function _seeded_rng_like(::Union{Random.TaskLocalRNG,Xoshiro}, ::AbstractComputeUnit, seeds::NTuple{4,UInt64})
    return Xoshiro(_nonzero_xoshiro_state(seeds)...)
end

# Xoshiro never leaves the all-zero state:
function _nonzero_xoshiro_state(seeds::NTuple{4,UInt64})
    return all(iszero, seeds) ? (UInt64(1), UInt64(0), UInt64(0), UInt64(0)) : seeds
end

function _seeded_rng_like(rng::AbstractRNG, cunit::AbstractComputeUnit, ::NTuple{4,UInt64})
    return throw(ArgumentError("Don't know how to create a new RNG like $(typeof(rng)) for compute unit $cunit"))
end


for (randfun, randfun!) in ((:rand, :rand!), (:randn, :randn!), (:randexp, :randexp!))
    @eval begin
        Random.$randfun(ctx::GenContext{T}) where T =
            _within_unit(() -> _draw_scalar(Random.$randfun!, ctx.rng, T, ctx.cunit), ctx.cunit)
        Random.$randfun(ctx::GenContext{T}, dims::Dims) where T = Random.$randfun!(ctx, allocate_array(ctx, dims))
        Random.$randfun(ctx::GenContext, dim1::Integer, dims::Integer...) = Random.$randfun(ctx, Dims((dim1, dims...)))
        Random.$randfun!(ctx::GenContext, A::AbstractArray) =
            _within_unit(() -> _fill_random!(Random.$randfun!, ctx.rng, A), ctx.cunit)
    end
end

# Random number generation rules, first specific to the RNG, then specific
# to the array:
_fill_random!(f!::F, rng::AbstractRNG, A::AbstractArray) where {F} = _fill_random_array!(f!, rng, A)

function _fill_random_array!(f!::F, rng::AbstractRNG, A::AbstractArray) where {F}
    if !(A isa GPUArraysCore.AnyGPUArray) && Adapt.unwrap_type(typeof(A)) <: GPUArraysCore.AbstractGPUArray
        # Native fills recognize only one layer of device array wrappers.
        tmp = similar(A, eltype(A), size(A))
        return A .= _fill_random_array!(f!, rng, tmp)
    end
    return f!(rng, A)
end
# GPU RNGs provide no exponential variates:
function _fill_random_array!(::typeof(Random.randexp!), rng::AbstractRNG, A::GPUArraysCore.AnyGPUArray)
    eltype(A) <: AbstractFloat || throw(ArgumentError("Exponential draws require a floating-point destination"))
    return _randexp_from_rand!(rng, A)
end

function _draw_scalar(f!::F, rng::AbstractRNG, ::Type{T}, cunit::AbstractComputeUnit) where {F,T}
    return is_host_unit(cunit) ? _scalar_randfun(f!)(rng, T)::T : _draw_scalar_via_array(f!, rng, T, cunit)
end

# GPU RNGs typically don't support scalar draws:
function _draw_scalar_via_array(f!::F, rng::AbstractRNG, ::Type{T}, cunit::AbstractComputeUnit) where {F,T}
    A = _fill_random!(f!, rng, allocate_array(cunit, T, (1,)))
    return only(Array(A))::T
end

_scalar_randfun(::typeof(Random.rand!)) = Random.rand
_scalar_randfun(::typeof(Random.randn!)) = Random.randn
_scalar_randfun(::typeof(Random.randexp!)) = Random.randexp

# Exponential variates from uniform variates in either [0, 1) or (0, 1]:
_randexp_from_rand!(rng::AbstractRNG, A::AbstractArray) = (A .= _neglog_uniform.(Random.rand!(rng, A)))
@inline _neglog_uniform(u::Real) = -log(ifelse(iszero(u), one(u), u))

function _randexp_from_rand!(rng::AbstractRNG, A::AbstractArray{Float16})
    bits = Random.rand!(rng, similar(A, UInt32))
    return A .= _float16_randexp.(bits)
end

# Use the midpoints of 2^32 uniform bins. Measuring from the nearer endpoint
# keeps both tails in range on devices that only support Float32 arithmetic.
@inline function _float16_randexp(bits::UInt32)
    u = (Float32(min(bits, ~bits)) + 0.5f0) * Float32(0x1p-32)
    return Float16(bits < 0x80000000 ? -log1p(-u) : -log(u))
end


"""
    allocate_array(ctx::GenContext, dims::Dims)
    allocate_array(ctx::GenContext, dims::Integer...)
    allocate_array(ctx::GenContext, ::Type{T}, dims::Dims)
    allocate_array(ctx::GenContext, ::Type{T}, dims::Integer...)

Allocate a new array on the compute unit and with the
numerical element type specified by `ctx`.

The default element type can be overriden by specifying `T`.
"""
@inline allocate_array(ctx::GenContext{T}, dims::Dims) where T = allocate_array(ctx.cunit, T, dims)
@inline allocate_array(ctx::GenContext, dims::Integer...) = allocate_array(ctx, Dims(dims))
@inline allocate_array(ctx::GenContext, ::Type{T}, args...) where T = allocate_array(ctx.cunit, T, args...)
@inline allocate_array(ctx::GenContext, ::Type{T}, dims::Integer...) where T = allocate_array(ctx, T, Dims(dims))


"""
    fill_array(ctx::GenContext, x, dims::Dims)
    fill_array(ctx::GenContext, x, dims::Integer...)

Create an array of size `dims` on the compute unit of `ctx`, filled with `x`.

Floating point values are converted to the precision of `ctx`.
"""
@inline fill_array(ctx::GenContext{T}, x, dims::Dims) where T = fill_array(ctx.cunit, _gen_convert(T, x), dims)
@inline fill_array(ctx::GenContext, x, dims::Integer...) = fill_array(ctx, x, Dims(dims))

_gen_convert(::Type{T}, x::AbstractFloat) where T = convert(T, x)
_gen_convert(::Type, x) = x
