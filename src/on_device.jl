# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

"""
    on_device(f, device::AbstractDevice, dummy_args...)

Returns a function that runs `f` on the specified `device` with arguments
like `dummy_args`.

The resulting function will only accept the same kind and number of arguments
as `dummy_args`. It will automatically adapt the arguments to the target
device, and adapt the function result back to the original device of the
arguments. Depending on the device, `dummy_args` may or may not be used.

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

function on_device(f, device::AbstractDevice, @nospecialize(dummy_args::Vararg{Any,N})) where {N}
    f_device = adapt(device, f)
    lock = nothing
    return _OnDevice{N}(f_device, device, lock)
end


struct _OnDevice{N,F,D,L<:Union{Nothing,ReentrantLock}} <: Function
    f_device::F
    device::D
    lock::L
end

function _OnDevice{N}(f_device::F, device::D, lock::L) where {N,F,D,L<:Union{Nothing,ReentrantLock}}
    return _OnDevice{N,F,D,L}(f_device, device, lock)
end


function (f::_OnDevice{N})(args::Vararg{Any,N}) where {N}
    dev_orig = get_device(args)
    adapted_args = adapt(f.device, args)
    result = _run_maybe_with_lock(f.f_device, f.lock, adapted_args...)
    readapted_result = adapt(dev_orig, result)
    return readapted_result
end

function (f::_OnDevice{N})(@nospecialize(args...)) where {N}
    return throws(
        ArgumentError("on_device function was created for $N arguments, can't handle $(length(args)) arguments")
    )
end


@inline _run_maybe_with_lock(f, ::Nothing, args::Vararg{Any,N}) where {N} = f(args...)
@inline _run_maybe_with_lock(f, lock::AbstractLock, args::Vararg{Any,N}) where {N} = @lock lock f(args...)
