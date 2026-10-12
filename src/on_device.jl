# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

"""
    on_device(f, cunit::AbstractComputeUnit, dummy_args...)
    on_device(f, device::MLDataDevices.AbstractDevice, dummy_args...)

Returns a function that runs `f` on the compute unit `cunit` (or the compute
unit of `device`) with arguments like `dummy_args`.

The resulting function will only accept the same kind and number of arguments
as `dummy_args`. It will automatically adapt the arguments to the target
compute unit, and adapt the function result back to the compute unit of the
arguments. Depending on the compute unit, `dummy_args` may or may not be used.
Arguments on different compute units are not supported.

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
    cunit_orig = _check_orig_cunit(get_compute_unit(args))
    adapted_args = adapt(f.cunit, args)
    result = _run_maybe_with_lock(f.f_device, f.lock, adapted_args...)
    readapted_result = adapt(cunit_orig, result)
    return readapted_result
end

function (f::_OnDevice{N})(@nospecialize(args...)) where {N}
    return throw(
        ArgumentError("on_device function was created for $N arguments, can't handle $(length(args)) arguments")
    )
end


_check_orig_cunit(cunit) = cunit
function _check_orig_cunit(::MixedComputeSystem)
    return throw(ArgumentError("on_device function arguments are on different compute units"))
end

@inline _run_maybe_with_lock(f, ::Nothing, args::Vararg{Any,N}) where {N} = f(args...)
@inline _run_maybe_with_lock(f, lock::AbstractLock, args::Vararg{Any,N}) where {N} = @lock lock f(args...)
