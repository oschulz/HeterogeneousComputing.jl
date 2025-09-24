module HeterogeneousComputingReactantExt

using HeterogeneousComputing
using HeterogeneousComputing: _OnDevice

using Reactant: @compile

using Adapt: adapt
using MLDataDevices: ReactantDevice


function HeterogeneousComputing.on_device(f, device::ReactantDevice, dummy_args::Vararg{Any,N}) where {N}
    f_dev, args_dev = adapt(device, (f, dummy_args))
    f_compiled = @compile f_dev(args_dev...)
    # Reactant-compiled functions are probably not thread-safe:
    lock = ReentrantLock()
    return _OnDevice{N}(f_dev, f_compiled, device, lock)
end


end # module HeterogeneousComputingReactantExt
