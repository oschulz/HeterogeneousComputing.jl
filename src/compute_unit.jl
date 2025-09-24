# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).


"""
    abstract type AbstractComputeUnit

Supertype for arbitrary compute units (CPU, GPU, etc.).

`adapt(cunit::AbstractComputeUnit, x)` adapts `x` for `cunit`.

`get_total_memory(cunit)` and `get_free_memory(cunit)` return the total
resp. the free memory on the compute unit.

[`allocate_array(cunit, dims)`](@ref) can be used to allocate new arrays
on `cunit`.

`KernelAbstractions.Backend(cunit)` will return default
[KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl)
backend for the type of the compute unit.

See also [`GenContext`](@ref).
"""
abstract type AbstractComputeUnit end
export AbstractComputeUnit


"""
    get_total_memory(cunit::AbstractComputeUnit)

Get the total amount of memory available on `cunit`.
"""
function get_total_memory end
export get_total_memory


"""
get_free_memory(cunit::AbstractComputeUnit)

Get the amount of free memory available on `cunit`.
"""
function get_free_memory end
export get_free_memory


"""
    HeterogeneousComputing.ka_backend(cunit::AbstractComputeUnit)

Returns the KernelAbstractions backend for `cunit`.

Requires KernelAbstractions.jl to be loaded, otherwise `ka_backend`
will have no methods.

Do not call directly, use for specialization only.

User code should call `KernelAbstractions.Backend(cunit)` or
`convert(KernelAbstractions.Backend, cunit)` instead, both of which
will use `ka_backend` internally.
"""
function ka_backend end



"""
    struct ComputeUnitIndependent

`get_compute_unit(x) === ComputeUnitIndependent()` indicates
that `x` is not tied to a specific compute unit. This typically
means that x is a statically allocated object.
"""
struct ComputeUnitIndependent end
export ComputeUnitIndependent


"""
    UnknownComputeUnitOf(x)

`get_compute_unit(x) === ComputeUnitIndependent()` indicates
that the compute unit for `x` cannot be determined.
"""
struct UnknownComputeUnitOf{T}
    x::T
end


"""
    struct MixedComputeSystem <: AbstractComputeUnit

A (possibly heterogenous) system of multiple compute units.
"""
struct MixedComputeSystem <: AbstractComputeUnit end
export MixedComputeSystem


"""
    merge_compute_units(compute_units...)

Merge `compute_units` unto a common/combined compute unit.

Do not specialize `merge_compute_units` directly,
specialize `compute_unit_mergerule(a, b)` instead.
"""
function merge_compute_units end
export merge_compute_units

merge_compute_units() = ComputeUnitIndependent()

@inline function merge_compute_units(a, b, c, ds::Vararg{Any,N}) where N
    a_b = merge_compute_units(a, b)
    return merge_compute_units(a_b, c, ds...)
end

@inline merge_compute_units(a::UnknownComputeUnitOf, b::UnknownComputeUnitOf) = a
@inline merge_compute_units(a::UnknownComputeUnitOf, b::Any) = a
@inline merge_compute_units(a::Any, b::UnknownComputeUnitOf) = b

@inline function merge_compute_units(a, b)
    return (a === b) ? a : compute_unit_mergeresult(
        compute_unit_mergerule(a, b),
        compute_unit_mergerule(b, a)
    )
end

struct NoCUnitMergeRule end

@inline compute_unit_mergerule(a::Any, b::Any) = NoCUnitMergeRule()
@inline compute_unit_mergerule(a::UnknownComputeUnitOf, b::Any) = a
@inline compute_unit_mergerule(a::UnknownComputeUnitOf, b::UnknownComputeUnitOf) = a
@inline compute_unit_mergerule(a::ComputeUnitIndependent, b::Any) = b

@inline compute_unit_mergeresult(a_b::NoCUnitMergeRule, b_a::NoCUnitMergeRule) = MixedComputeSystem()
@inline compute_unit_mergeresult(a_b, b_a::NoCUnitMergeRule) = a_b
@inline compute_unit_mergeresult(a_b::NoCUnitMergeRule, b_a) = b_a
@inline compute_unit_mergeresult(a_b, b_a) = a_b === b_a ? a_b : MixedComputeSystem()


"""
    get_compute_unit(x)::Union{
        AbstractComputeUnit,
        ComputeUnitIndependent,
        UnknownComputeUnitOf
    }

Get the compute unit backing object `x`.

Don't specialize `get_compute_unit`, specialize
[`HeterogeneousComputing.get_compute_unit_impl`](@ref) instead.
"""
function get_compute_unit end
export get_compute_unit

get_compute_unit(x) = get_compute_unit_impl(Union{}, x)
get_compute_unit(cunit::AbstractComputeUnit) = cunit


"""
    HeterogeneousComputing.get_compute_unit_impl(::Type{TypeHistory}, x)::AbstractComputeUnit

See [`get_compute_unit_impl`](@ref).

Specializations that directly resolve the compute unit based on `x` can
ignore `TypeHistory`:

```julia
HeterogeneousComputing.get_compute_unit_impl(@nospecialize(TypeHistory::Type), x::SomeType) = ...
```
"""
function get_compute_unit_impl end


# Guard against object reference loops:
@inline get_compute_unit_impl(::Type{TypeHistory}, x::T) where {TypeHistory,T<:TypeHistory} = begin
    UnknownComputeUnitOf(x)
end

@generated function get_compute_unit_impl(::Type{TypeHistory}, x) where TypeHistory
    if isbitstype(x)
        :(ComputeUnitIndependent())
    else
        NewTypeHistory = Union{TypeHistory,x}
        impl = :(
            begin
                dev_0 = ComputeUnitIndependent()
            end
        )
        append!(
            impl.args,
            [
                :(
                    $(Symbol(:dev_, i)) = merge_compute_units(
                        get_compute_unit_impl($NewTypeHistory, getfield(x, $i)),
                        $(Symbol(:dev_, i-1))
                    )
                ) for i in 1:fieldcount(x)
            ]
        )
        push!(impl.args, :(return $(Symbol(:dev_, fieldcount(x)))))
        impl
    end
end



"""
    struct CPUnit <: AbstractComputeUnit

`CPUnit()` is the default central processing unit (CPU).
"""
struct CPUnit <: AbstractComputeUnit end
export CPUnit

Adapt.adapt_storage(::CPUnit, x::AbstractArray) = Adapt.adapt(Array, x)

get_total_memory(::CPUnit) = Sys.total_memory()
get_free_memory(::CPUnit) = Sys.free_memory()

@inline get_compute_unit_impl(@nospecialize(TypeHistory::Type), ::Array) = CPUnit()



"""
    abstract type AbstractComputeAccelerator <: AbstractComputeUnit

Supertype for GPU compute units.
"""
abstract type AbstractComputeAccelerator <: AbstractComputeUnit end
export AbstractComputeAccelerator


"""
    abstract type AbstractGPUnit <: AbstractComputeAccelerator

Supertype for GPU comute units.
"""
abstract type AbstractGPUnit <: AbstractComputeAccelerator end
export AbstractGPUnit


"""
    allocate_array(cpunit::AbstractComputeUnit, ::Type{T}, dims::Dims)
    allocate_array(cpunit::AbstractComputeUnit, ::Type{T}, dims::Integer...)

Allocate a new array with element type `T` and size `dims` on compute unit
`cunit`.

The content of the newly allocated array is undefined.
"""
function allocate_array end
export allocate_array

allocate_array(cpunit::AbstractComputeUnit, ::Type{T}, dims::Integer...) where T = allocate_array(cpunit, T, dims)

allocate_array(::CPUnit, ::Type{T}, dims::Dims) where T = Array{T}(undef, dims)
