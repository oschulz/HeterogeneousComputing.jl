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
uses `CPUnit()` for host RNGs.

When constructed from an `MLDataDevices` device without an explicit `T`, the
precision is `eltype(dev)` if that is a floating point type, `Float64`
otherwise.

`get_precision(ctx)`, `get_compute_unit(ctx)` and `get_rng(ctx)` return the
precision, compute unit and random number generator of the context.

`rand(ctx, dims)`, `randn(ctx, dims)` and `randexp(ctx, dims)` generate arrays
of random numbers with precision `T` on the compute unit of `ctx`,
`rand!(ctx, A)`, `randn!(ctx, A)` and `randexp!(ctx, A)` fill `A` (which
should reside on the compute unit of `ctx`) with random numbers.
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
GenContext{T}(rng::AbstractRNG) where T = GenContext{T}(_rng_cunit(get_compute_unit(rng)), rng)

_rng_cunit(cunit::AbstractComputeUnit) = cunit
_rng_cunit(::Any) = CPUnit()

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

# Adapting a context to a compute unit or device moves it to that unit:
function Adapt.adapt_structure(to, ctx::GenContext{T}) where T
    return GenContext{T}(_adapted_cunit(to, ctx.cunit), adapt(to, ctx.rng))
end

_adapted_cunit(to::AbstractComputeUnit, ::AbstractComputeUnit) = to
_adapted_cunit(to::AbstractDevice, ::AbstractComputeUnit) = DeviceUnit(to)
_adapted_cunit(@nospecialize(to), cunit::AbstractComputeUnit) = cunit


for (randfun, randfun!) in ((:rand, :rand!), (:randn, :randn!), (:randexp, :randexp!))
    @eval begin
        Random.$randfun(ctx::GenContext{T}) where T =
            _within_unit(() -> _draw_scalar(Random.$randfun!, ctx.rng, T, ctx.cunit), ctx.cunit)
        Random.$randfun(ctx::GenContext{T}, dims::Dims) where T = Random.$randfun!(ctx, allocate_array(ctx, dims))
        Random.$randfun(ctx::GenContext, dim1::Integer, dims::Integer...) = Random.$randfun(ctx, (dim1, dims...))
        Random.$randfun!(ctx::GenContext, A::AbstractArray) =
            _within_unit(() -> _fill_random!(Random.$randfun!, ctx.rng, A), ctx.cunit)
    end
end

# Random number generation rules, first specific to the RNG, then specific
# to the array:
_fill_random!(f!::F, rng::AbstractRNG, A::AbstractArray) where {F} = _fill_random_array!(f!, rng, A)

_fill_random_array!(f!::F, rng::AbstractRNG, A::AbstractArray) where {F} = f!(rng, A)
# GPU RNGs provide no exponential variates:
function _fill_random_array!(::typeof(Random.randexp!), rng::AbstractRNG, A::GPUArraysCore.AbstractGPUArray)
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
@inline allocate_array(ctx::GenContext, dim1::Integer, dims::Integer...) = allocate_array(ctx, (dim1, dims...))
@inline allocate_array(ctx::GenContext, ::Type{T}, args...) where T = allocate_array(ctx.cunit, T, args...)
@inline allocate_array(ctx::GenContext, ::Type{T}, dims::Integer...) where T = allocate_array(ctx, T, dims)


"""
    fill_array(ctx::GenContext, x, dims::Dims)
    fill_array(ctx::GenContext, x, dims::Integer...)

Create an array of size `dims` on the compute unit of `ctx`, filled with `x`.

Floating point values are converted to the precision of `ctx`.
"""
@inline fill_array(ctx::GenContext{T}, x, dims::Dims) where T = fill_array(ctx.cunit, _gen_convert(T, x), dims)
@inline fill_array(ctx::GenContext, x, dims::Integer...) = fill_array(ctx, x, dims)

_gen_convert(::Type{T}, x::AbstractFloat) where T = convert(T, x)
_gen_convert(::Type, x) = x
