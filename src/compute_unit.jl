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
resp. the free memory on the compute unit (currently only supported for the
CPU, CUDA and JLArrays).

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

function MLDataDevices.default_device_rng(::MixedComputeSystem)
    return throw(ArgumentError("A MixedComputeSystem has no default random number generator"))
end


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

Unless there is a specific rule for objects of type `typeof(x)` (see
[`HeterogeneousComputing.get_compute_unit_impl`](@ref)), `get_compute_unit`
recurses through the fields of `x` and merges their compute units via
[`merge_compute_units`](@ref). Reference loops are handled.

The compute units of GPU arrays and random number generators are based on
`MLDataDevices.get_device`.

Don't specialize `get_compute_unit`, specialize
[`HeterogeneousComputing.get_compute_unit_impl`](@ref) instead.
"""
function get_compute_unit end
export get_compute_unit

get_compute_unit(x) = _get_cunit(x, nothing)


"""
    HeterogeneousComputing.get_compute_unit_impl(x)

Specialize `get_compute_unit_impl(x::SomeType)` to directly return the
compute unit of objects of type `SomeType`, instead of recursing through
their fields.

See [`get_compute_unit`](@ref).
"""
function get_compute_unit_impl end

struct _RecurseFields end

get_compute_unit_impl(@nospecialize(x)) = _RecurseFields()
get_compute_unit_impl(cunit::AbstractComputeUnit) = cunit
get_compute_unit_impl(@nospecialize(T::Type)) = ComputeUnitIndependent()


# `visited` is `nothing` or an `IdSet` of visited objects that may be part of
# reference loops:
@inline _get_cunit(x, visited) = _get_cunit_via(get_compute_unit_impl(x), x, visited)

@inline _get_cunit_via(cunit, @nospecialize(x), @nospecialize(visited)) = cunit
@inline _get_cunit_via(::_RecurseFields, x, visited) = _get_fields_cunit(x, visited)

@generated function _get_fields_cunit(x, visited)
    isbitstype(x) && return :(ComputeUnitIndependent())
    may_loop = _may_close_ref_loop(x)
    field_cunit(i) = :(_get_cunit(getfield(x, $i), visited))
    field_cunit_maybe_undef(i) = :(isdefined(x, $i) ? $(field_cunit(i)) : ComputeUnitIndependent())
    body = Expr(:block)
    if may_loop
        push!(body.args, :(visited = _visit!(visited, x)))
        push!(body.args, :(visited isa _AlreadyVisited && return ComputeUnitIndependent()))
    end
    push!(body.args, :(cunit_0 = ComputeUnitIndependent()))
    for i in 1:fieldcount(x)
        fcunit = ismutabletype(x) && !isbitstype(fieldtype(x, i)) ? field_cunit_maybe_undef(i) : field_cunit(i)
        push!(body.args, :($(Symbol(:cunit_, i)) = merge_compute_units($fcunit, $(Symbol(:cunit_, i - 1)))))
    end
    push!(body.args, :(return $(Symbol(:cunit_, fieldcount(x)))))
    return body
end

# Whether objects of type T may be part of a reference loop. Loops need a
# mutable object that can reach itself: either via fields of non-concrete
# type or via concretely typed fields only.
function _may_close_ref_loop(@nospecialize(T::Type))
    ismutabletype(T) || return false
    seen = Set{Any}()
    pending = Any[fieldtypes(T)...]
    while !isempty(pending)
        FT = pop!(pending)
        isbitstype(FT) && continue
        (!isconcretetype(FT) || FT === T) && return true
        FT in seen && continue
        push!(seen, FT)
        append!(pending, fieldtypes(FT))
    end
    return false
end

struct _AlreadyVisited end

_visit!(::Nothing, x) = _visit!(IdSet{Any}(), x)
_visit!(visited::IdSet{Any}, x) = x in visited ? _AlreadyVisited() : push!(visited, x)


@inline get_compute_unit_impl(A::GPUArraysCore.AbstractGPUArray) = _gpu_array_cunit(get_device(A), A)

_gpu_array_cunit(dev::AbstractDevice, @nospecialize(A)) = DeviceUnit(dev)
# MLDataDevices considers arrays of unknown type to be CPU arrays:
_gpu_array_cunit(::Union{CPUDevice,UnknownDevice,Nothing}, A) = UnknownComputeUnitOf(A)

# RNGs without a known device may still hold device state:
@inline get_compute_unit_impl(rng::AbstractRNG) = _rng_cunit(get_device(rng))

_rng_cunit(dev::AbstractDevice) = DeviceUnit(dev)
_rng_cunit(::Union{UnknownDevice,Nothing}) = _RecurseFields()



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
device (e.g. `CUDADevice()`) is resolved to that specific device. Reactant
units also drop sharding and number tracking settings, like element types
these are data movement policies, not properties of a compute unit. So units
constructed from devices and units derived from data via
[`get_compute_unit`](@ref) are equal.

Throws an `ArgumentError` if the package that provides the device isn't
loaded.

`MLDataDevices.get_device(cunit)` returns the device.
"""
struct DeviceUnit{D<:AbstractDevice} <: AbstractComputeUnit
    device::D

    function DeviceUnit(dev::AbstractDevice)
        _device_loaded(dev) || throw(ArgumentError("Package for device $dev is not loaded"))
        canonical_dev = _canonical_device(dev)
        return new{typeof(canonical_dev)}(canonical_dev)
    end
end
export DeviceUnit

_device_loaded(dev::AbstractDevice) = MLDataDevices.loaded(dev)

_canonical_device(dev::AbstractDevice) = dev

const _EltypeDevice = Union{CPUDevice,CUDADevice,AMDGPUDevice,MetalDevice,oneAPIDevice,OpenCLDevice,ReactantDevice}
_canonical_device(dev::_EltypeDevice) = MLDataDevices.with_eltype(dev, nothing)

Base.show(io::IO, cunit::DeviceUnit) = print(io, "DeviceUnit(", cunit.device, ")")

AbstractComputeUnit(dev::AbstractDevice) = DeviceUnit(dev)
Base.convert(::Type{AbstractComputeUnit}, dev::AbstractDevice) = DeviceUnit(dev)

MLDataDevices.get_device(cunit::DeviceUnit) = cunit.device
MLDataDevices.default_device_rng(cunit::DeviceUnit) = _within_unit(() -> default_device_rng(cunit.device), cunit)

Adapt.adapt_storage(cunit::DeviceUnit, x) = Adapt.adapt_storage(cunit.device, x)

# Devices may contain handles that are equal but not identical, so units are
# compared via device equality and hashed by device kind:
Base.:(==)(a::DeviceUnit, b::DeviceUnit) = _same_device(a.device, b.device)
Base.hash(cunit::DeviceUnit, h::UInt) = hash(Base.typename(typeof(cunit.device)), hash(DeviceUnit, h))

_same_device(a::AbstractDevice, b::AbstractDevice) = a == b

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

@inline get_compute_unit_impl(::Array) = CPUnit()
@static if isdefined(Core, :Memory)
    @inline get_compute_unit_impl(::Memory) = CPUnit()
end



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

# Run `f` with `cunit` as the active device of its backend (for backends
# with a notion of an active device):
_within_unit(f, @nospecialize(cunit::AbstractComputeUnit)) = f()

# Generic fallback, avoids transferring data from the host:
function allocate_array(cunit::DeviceUnit, ::Type{T}, dims::Dims) where T
    return _within_unit(() -> similar(adapt(cunit, Vector{T}()), dims), cunit)
end


"""
    fill_array(cunit::AbstractComputeUnit, x, dims::Dims)
    fill_array(cunit::AbstractComputeUnit, x, dims::Integer...)

Create an array of size `dims` on compute unit `cunit`, filled with `x`.
"""
function fill_array end
export fill_array

function fill_array(cunit::AbstractComputeUnit, x, dims::Dims)
    return _within_unit(() -> fill!(allocate_array(cunit, typeof(x), dims), x), cunit)
end
fill_array(cunit::AbstractComputeUnit, x, dims::Integer...) = fill_array(cunit, x, dims)
