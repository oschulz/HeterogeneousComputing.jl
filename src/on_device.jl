# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

"""
    on_device(f, cunit::AbstractComputeUnit, dummy_args...)
    on_device(f, device::MLDataDevices.AbstractDevice, dummy_args...)

Returns a function that runs `f` on the compute unit `cunit` (or the compute
unit of `device`) with arguments like `dummy_args`.

The resulting function takes the same number of arguments as `dummy_args`.
Compiling backends such as Reactant can also specialize on argument types
and array sizes. For those backends, call `on_device` again when the types
or sizes change.

The function adapts its arguments to `cunit` and the result back to the
compute unit of the arguments, or leaves the result on `cunit` if the
arguments aren't tied to any compute unit. Arguments on different compute
units are not supported.

Example:

```julia
using HeterogeneousComputing, MLDataDevices, Reactant

f(x, y) = sum(x .* y)
dummy_x, dummy_y = rand(Float32, 10), rand(Float32, 10)

device = ReactantDevice()
g = on_device(f, device, dummy_x, dummy_y)

x, y = rand(Float32, 10), rand(Float32, 10)
g(x, y) ≈ f(x, y)
```
"""
function on_device end
export on_device

function on_device(f, cunit::AbstractComputeUnit, @nospecialize(dummy_args::Vararg{Any,N})) where {N}
    f_device = adapt(cunit, f)
    lock = nothing
    return _OnDevice{N}(f_device, cunit, lock)
end

function on_device(f, device::AbstractDevice, @nospecialize(dummy_args::Vararg{Any,N})) where {N}
    return on_device(f, DeviceUnit(device), dummy_args...)
end


struct _OnDevice{N,F,CU<:AbstractComputeUnit,L<:Union{Nothing,ReentrantLock}} <: Function
    f_device::F
    cunit::CU
    lock::L
end

function _OnDevice{N}(f_device::F, cunit::CU, lock::L) where {N,F,CU,L<:Union{Nothing,ReentrantLock}}
    return _OnDevice{N,F,CU,L}(f_device, cunit, lock)
end


function (f::_OnDevice{N})(args::Vararg{Any,N}) where {N}
    result_cunit = _result_cunit(args, f.cunit)
    adapted_args = adapt(f.cunit, args)
    result = _within_unit(() -> _run_maybe_with_lock(f.f_device, f.lock, adapted_args...), f.cunit)
    return adapt(result_cunit, result)
end

function (f::_OnDevice{N})(@nospecialize(args...)) where {N}
    return throw(
        ArgumentError("on_device function was created for $N arguments, can't handle $(length(args)) arguments")
    )
end


struct _ArgumentUnits{K,U}
    known::K
    unknown::U
end
_ArgumentUnits(unit) = _ArgumentUnits(unit, nothing)
_ArgumentUnits(unit::UnknownComputeUnitOf) = _ArgumentUnits(ComputeUnitIndependent(), unit)
_ArgumentUnits(units::_ArgumentUnits) = units
function _ArgumentUnits(::MixedComputeSystem, unknown)
    return throw(ArgumentError("on_device function arguments are on different compute units"))
end

# An unknown result adaptor can coexist with known units deeper in the arguments.
function _merge_argument_units(a, b)
    left, right = _ArgumentUnits(a), _ArgumentUnits(b)
    known = merge_compute_units(left.known, right.known)
    return _ArgumentUnits(known, isnothing(left.unknown) ? right.unknown : left.unknown)
end

function _result_cunit(args::Tuple, cunit::AbstractComputeUnit)
    units = _ArgumentUnits(_get_fields_cunit(args, nothing, _merge_argument_units))
    isnothing(units.unknown) || return units.unknown
    return units.known isa ComputeUnitIndependent ? cunit : units.known
end

@inline _run_maybe_with_lock(f, ::Nothing, args::Vararg{Any,N}) where {N} = f(args...)
@inline _run_maybe_with_lock(f, lock::AbstractLock, args::Vararg{Any,N}) where {N} = @lock lock f(args...)
