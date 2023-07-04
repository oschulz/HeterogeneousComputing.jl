# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).


"""
    GenContext{T=AbstractFloat}(
        rng::AbstractRNG = Random.default_rng(),
        cunit::AbstractComputeUnit = CPUnit()
    ) <: AbstractRNG

Context for generative computations.

* `Base.eltype(ctx::GenContext)`` will return `T`.
"""
struct GenContext{T<:AbstractFloat,RNG<:AbstractRNG,CU<:AbstractComputeUnit} <: AbstractRNG
    rng::RNG
    cunit::CU
end

export GenContext

@inline GenContext{T,RNG,CU}(ctx::GenContext) where {T,RNG,CU} = GenContext{T,RNG,CU}(ctx.rng, ctx.cunit)
@inline GenContext{T}(ctx::GenContext) where T = GenContext{T}(ctx.rng, ctx.cunit)
@inline GenContext(ctx::GenContext{T}) where T = GenContext{T}(ctx.rng, ctx.cunit)
Base.convert(::Type{GenContext{T}}, ctx::GenContext) where T = GenContext{T}(ctx)

Base.eltype(ctx::GenContext{T}) where T = T

@inline GenContext(args...) = GenContext{Float64}(args...)

@inline function GenContext{T}(rng::RNG = Random.default_rng(), cpunit::CU = CPUnit()) where {T,RNG,CU}
    GenContext{T,RNG,CU}(rng, cpunit)
end

GenContext{T}(cunit::AbstractComputeUnit) where T = GenContext{T}(Random.default_rng(), cunit)

for (randfun, randfun!) in ((:rand,:rand!), (:randn,:randn!), (:randexp,:randexp!))
    @eval begin
        Random.$randfun(ctx::GenContext{T}) where T = Random.$randfun(ctx.rng, T)
        Random.$randfun(ctx::GenContext, T::Random.BitFloatType) = Random.$randfun(ctx.rng, T)
        function Random.$randfun(ctx::GenContext{T}, dims::Dims) where T
            A = allocate_array(ctx.cunit, T, dims)
            Random.$randfun!(ctx.rng, A)
            return A
        end
        Random.$randfun(ctx::GenContext{T}, dims::Integer...) where T = Random.$randfun(ctx, dims)
        Random.$randfun(ctx::GenContext, T::Random.BitFloatType, dims::Dims) = Random.$randfun(ctx.rng, T, dims)

        Random.$randfun!(ctx::GenContext, A::AbstractArray{T}) where T = Random.$randfun!(ctx.rng, A)
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
