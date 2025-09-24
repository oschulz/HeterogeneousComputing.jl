module HeterogeneousComputingReactantExt

using HeterogeneousComputing
using HeterogeneousComputing: _OnDevice

using Reactant.Compiler: compile

using Adapt: adapt
using MLDataDevices: ReactantDevice


function HeterogeneousComputing.on_device(f, device::ReactantDevice, dummy_args::Vararg{Any,N}) where {N}
    f_device, args_device = adapt(device, (f, dummy_args))
    f_compiled = compile(f_device, args_device)
    # Reactant-compiled functions are probably not thread-safe:
    lock = ReentrantLock()
    return _OnDevice{N}(f_compiled, device, lock)
end


end # module HeterogeneousComputingReactantExt
