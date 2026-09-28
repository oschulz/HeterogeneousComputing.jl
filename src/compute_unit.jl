# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).


"""
    abstract type AbstractComputeUnit

Supertype for arbitrary compute units (CPU, GPU, etc.).

Most compute units are [`DeviceUnit`](@ref)s, which are based on
`MLDataDevices` devices. `AbstractComputeUnit(dev::MLDataDevices.AbstractDevice)`
returns the compute unit for a device.

`adapt(cunit::AbstractComputeUnit, x)` adapts `x` for `cunit`, without
changing the numerical precision of `x`.

`get_total_memory(cunit)` and `get_free_memory(cunit)` return the total
resp. the free memory on the compute unit.

[`allocate_array(cunit, T, dims)`](@ref) and [`fill_array(cunit, x, dims)`](@ref)
can be used to create new arrays on `cunit`.

`MLDataDevices.default_device_rng(cunit)` returns the default random number
generator for `cunit`.

`KernelAbstractions.Backend(cunit)` will return the default
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
    is_host_unit(cunit::AbstractComputeUnit)::Bool

Whether arrays on `cunit` can be processed by generic host code, e.g. via
scalar indexing.
"""
function is_host_unit end
export is_host_unit

is_host_unit(::AbstractComputeUnit) = false


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

`get_compute_unit(x) === UnknownComputeUnitOf(x)` indicates
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
    return _same_cunit(a, b) ? a : compute_unit_mergeresult(
        compute_unit_mergerule(a, b),
        compute_unit_mergerule(b, a)
    )
end

@inline _same_cunit(a, b) = a === b

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

Recurses through the fields of `x` and merges the compute units of the
leaves via [`merge_compute_units`](@ref). The compute units of GPU arrays and
random number generators are based on `MLDataDevices.get_device`.

Don't specialize `get_compute_unit`, specialize
[`HeterogeneousComputing.get_compute_unit_impl`](@ref) instead.
"""
function get_compute_unit end
export get_compute_unit

get_compute_unit(x) = get_compute_unit_impl(Union{}, x)
get_compute_unit(cunit::AbstractComputeUnit) = cunit


"""
    HeterogeneousComputing.get_compute_unit_impl(::Type{TypeHistory}, x)

See [`get_compute_unit`](@ref).

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
                        $(Symbol(:dev_, i - 1))
                    )
                ) for i in 1:fieldcount(x)
            ]
        )
        push!(impl.args, :(return $(Symbol(:dev_, fieldcount(x)))))
        impl
    end
end

@inline get_compute_unit_impl(@nospecialize(TypeHistory::Type), A::GPUArraysCore.AbstractGPUArray) =
    _cunit_from_device(get_device(A), A)

@inline get_compute_unit_impl(@nospecialize(TypeHistory::Type), rng::AbstractRNG) =
    _cunit_from_device(get_device(rng), rng)

_cunit_from_device(dev::AbstractDevice, @nospecialize(x)) = DeviceUnit(dev)
_cunit_from_device(::UnknownDevice, x) = UnknownComputeUnitOf(x)
_cunit_from_device(::Nothing, @nospecialize(x)) = ComputeUnitIndependent()



"""
    struct DeviceUnit{D<:MLDataDevices.AbstractDevice} <: AbstractComputeUnit

A compute unit based on an `MLDataDevices` device.

Constructors:

```julia
DeviceUnit(dev::MLDataDevices.AbstractDevice)
AbstractComputeUnit(dev::MLDataDevices.AbstractDevice)
```

The device is normalized on construction: The unit has no element type
(`adapt(cunit, x)` preserves numerical precision, use a [`GenContext`](@ref)
to specify precision), and a device that refers to the currently active
device (e.g. `CUDADevice()`) is resolved to that specific device. So units
constructed from devices and units derived from data via
[`get_compute_unit`](@ref) are equal.

`MLDataDevices.get_device(cunit)` returns the device.
"""
struct DeviceUnit{D<:AbstractDevice} <: AbstractComputeUnit
    device::D

    function DeviceUnit(dev::AbstractDevice)
        canonical_dev = _canonical_device(dev)
        return new{typeof(canonical_dev)}(canonical_dev)
    end
end
export DeviceUnit

_canonical_device(dev::AbstractDevice) = MLDataDevices.with_eltype(dev, nothing)

AbstractComputeUnit(dev::AbstractDevice) = DeviceUnit(dev)
Base.convert(::Type{AbstractComputeUnit}, dev::AbstractDevice) = DeviceUnit(dev)

MLDataDevices.get_device(cunit::DeviceUnit) = cunit.device
MLDataDevices.default_device_rng(cunit::DeviceUnit) = default_device_rng(cunit.device)

Adapt.adapt_storage(cunit::DeviceUnit, x) = Adapt.adapt_storage(cunit.device, x)

# Devices may contain handles that are equal but not identical, and devices
# derived within traced code may specify their location only partially
# (equal to any location), so units are compared via device equality and
# hashed by device kind only:
Base.:(==)(a::DeviceUnit, b::DeviceUnit) = a.device == b.device
Base.hash(cunit::DeviceUnit, h::UInt) = hash(Base.typename(typeof(cunit.device)), hash(DeviceUnit, h))

@inline _same_cunit(a::DeviceUnit, b::DeviceUnit) = a == b


"""
    const CPUnit = DeviceUnit{MLDataDevices.CPUDevice{Nothing}}

`CPUnit()` is the default central processing unit (CPU).
"""
const CPUnit = DeviceUnit{CPUDevice{Nothing}}
export CPUnit

(::Type{CPUnit})() = DeviceUnit(CPUDevice())

Base.show(io::IO, ::CPUnit) = print(io, "CPUnit()")

is_host_unit(::CPUnit) = true

get_total_memory(::CPUnit) = Sys.total_memory()
get_free_memory(::CPUnit) = Sys.free_memory()

@inline get_compute_unit_impl(@nospecialize(TypeHistory::Type), ::Array) = CPUnit()



"""
    allocate_array(cunit::AbstractComputeUnit, ::Type{T}, dims::Dims)
    allocate_array(cunit::AbstractComputeUnit, ::Type{T}, dims::Integer...)

Allocate a new array with element type `T` and size `dims` on compute unit
`cunit`.

The content of the newly allocated array is undefined.
"""
function allocate_array end
export allocate_array

allocate_array(cunit::AbstractComputeUnit, ::Type{T}, dims::Integer...) where T = allocate_array(cunit, T, dims)

allocate_array(::CPUnit, ::Type{T}, dims::Dims) where T = Array{T}(undef, dims)

# Generic fallback, avoids transferring data from the host:
function allocate_array(cunit::DeviceUnit, ::Type{T}, dims::Dims) where T
    return similar(adapt(cunit, Vector{T}()), dims)
end


"""
    fill_array(cunit::AbstractComputeUnit, x, dims::Dims)
    fill_array(cunit::AbstractComputeUnit, x, dims::Integer...)

Create an array of size `dims` on compute unit `cunit`, filled with `x`.
"""
function fill_array end
export fill_array

fill_array(cunit::AbstractComputeUnit, x, dims::Dims) = fill!(allocate_array(cunit, typeof(x), dims), x)
fill_array(cunit::AbstractComputeUnit, x, dims::Integer...) = fill_array(cunit, x, dims)
