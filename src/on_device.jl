# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

"""
    on_device(f, cunit::AbstractComputeUnit, dummy_args...)
    on_device(f, device::MLDataDevices.AbstractDevice, dummy_args...)

Returns a function that runs `f` on the compute unit `cunit` (or the compute
unit of `device`) with arguments like `dummy_args`.

The resulting function will only accept the same kind and number of arguments
as `dummy_args`. It will automatically adapt the arguments to the target
compute unit, and adapt the function result back to the compute unit of the
arguments (or to the target compute unit, if the arguments aren't tied to a
compute unit). Depending on the compute unit, `dummy_args` may or may not be
used. Arguments on different compute units are not supported.

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
    result_cunit = _result_cunit(get_compute_unit(args), f.cunit)
    adapted_args = adapt(f.cunit, args)
    result = _within_unit(() -> _run_maybe_with_lock(f.f_device, f.lock, adapted_args...), f.cunit)
    return adapt(result_cunit, result)
end

function (f::_OnDevice{N})(@nospecialize(args...)) where {N}
    return throw(
        ArgumentError("on_device function was created for $N arguments, can't handle $(length(args)) arguments")
    )
end


# Results go back to the compute unit of the arguments:
_result_cunit(cunit_args, @nospecialize(cunit)) = cunit_args
_result_cunit(::ComputeUnitIndependent, cunit::AbstractComputeUnit) = cunit
function _result_cunit(::MixedComputeSystem, @nospecialize(cunit))
    return throw(ArgumentError("on_device function arguments are on different compute units"))
end

@inline _run_maybe_with_lock(f, ::Nothing, args::Vararg{Any,N}) where {N} = f(args...)
@inline _run_maybe_with_lock(f, lock::AbstractLock, args::Vararg{Any,N}) where {N} = @lock lock f(args...)
