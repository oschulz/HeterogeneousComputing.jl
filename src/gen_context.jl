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
    GenContext{T=AbstractFloat}(
        rng::AbstractRNG = Random.default_rng(),
        cunit::AbstractComputeUnit = CPUnit()
    )

Context for generative computations.

* `Base.eltype(ctx::GenContext)`` will return `T`.
"""
struct GenContext{T<:AbstractFloat,CU<:AbstractComputeUnit,RNG<:AbstractRNG}
    cunit::CU
    rng::RNG
end

export GenContext

@inline GenContext{T}(cpunit::CU, rng::RNG) where {T,CU,RNG} = GenContext{T,CU,RNG}(cpunit, rng)

@inline GenContext(args...) = GenContext{Float64}(args...)
GenContext{T}() where T = GenContext{T}(CPUnit(), Random.default_rng())
GenContext{T}(cpunit::AbstractComputeUnit) where T = GenContext{T}(cpunit, Random.default_rng())
# ToDo: Derive cunit from RNG type, e.g. if RNG is a GPU-specific RNG?
GenContext{T}(rng::AbstractRNG) where T = GenContext{T}(CPUnit(), rng)

@inline GenContext{T,RNG,CU}(ctx::GenContext) where {T,RNG,CU} = GenContext{T,RNG,CU}(ctx.cunit, ctx.rng)
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

get_gencontext(x::T) where T = _generic_get_gencontext(T, get_precision(x), get_compute_unit(x), get_rng(x))

function _generic_get_gencontext(::TX, ::Type{T}, cunit::AbstractComputeUnit, rng::AbstractRNG) where {TX,T<:AbstractFloat}
    GenContext{T}(cunit, rng)
end

function _generic_get_gencontext(::TX, ::Type{T}, cunit::AbstractComputeUnit, rng::NoRNG) where {TX,T<:AbstractFloat}
    GenContext{T}(cunit)
end

_generic_get_gencontext(::TX, ::Type, ::Any, ::Any) where TX = NoGenContext{TX}()

get_gencontext(ctx::GenContext) = ctx


get_precision_fromtype(::Type{<:GenContext{T}}) where T = T
get_compute_unit(ctx::GenContext) = ctx.cunit
get_rng(ctx::GenContext) = ctx.rng

#Base.eltype(ctx::GenContext) = get_precision(ctx)
#Random.AbstractRNG(ctx::GenContext) = get_rng(ctx)
#AbstractComputeUnit(ctx::GenContext) = get_compute_unit(ctx)


for (randfun, randfun!) in ((:rand,:rand!), (:randn,:randn!), (:randexp,:randexp!))
    @eval begin
        Random.$randfun(ctx::GenContext{T}) where T = Random.$randfun(ctx.rng, T)
        function Random.$randfun(ctx::GenContext{T}, dims::Dims) where T
            A = allocate_array(ctx.cunit, T, dims)
            Random.$randfun!(ctx.rng, A)
            return A
        end
        Random.$randfun(ctx::GenContext{T}, dims::Integer...) where T = Random.$randfun(ctx, dims)

        # Don't support mutating rng functions for now.
        # Random.$randfun!(ctx::GenContext, A::AbstractArray{T}) where T = Random.$randfun!(ctx.rng, A)
    end
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
@inline allocate_array(ctx::GenContext, dims::Integer...) = allocate_array(ctx, dims)
@inline allocate_array(ctx::GenContext, ::Type{T}, args...) where T = allocate_array(ctx.cunit, T, args...)
@inline allocate_array(ctx::GenContext, ::Type{T}, dims::Integer...) where T = allocate_array(ctx, T, dims)
