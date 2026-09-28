# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingReactantExt

using HeterogeneousComputing
using HeterogeneousComputing: _OnDevice
import HeterogeneousComputing: get_compute_unit_impl, get_precision_fromtype, allocate_array, _canonical_device

import Reactant
using Reactant: ConcreteRArray, TracedRArray, TracedRNumber, ReactantRNG, within_compile
using Reactant.Compiler: compile

using Adapt: adapt
using MLDataDevices: MLDataDevices, ReactantDevice, get_device


const ReactantUnit = DeviceUnit{<:ReactantDevice}

const _ConcreteTypes = Union{
    Reactant.ConcretePJRTArray,Reactant.ConcretePJRTNumber,
    Reactant.ConcreteIFRTArray,Reactant.ConcreteIFRTNumber
}

# Units don't track the sharding of individual arrays:
function _canonical_device(dev::ReactantDevice)
    return MLDataDevices.with_eltype(ReactantDevice(dev.client, dev.device, missing), nothing)
end

get_compute_unit_impl(@nospecialize(TypeHistory::Type), x::_ConcreteTypes) = DeviceUnit(get_device(x))
# The device of traced values is unknown within compiled code:
function get_compute_unit_impl(@nospecialize(TypeHistory::Type), ::Union{TracedRArray,TracedRNumber})
    return DeviceUnit(ReactantDevice())
end
get_compute_unit_impl(TypeHistory::Type, rng::ReactantRNG) = get_compute_unit_impl(TypeHistory, rng.seed)

get_precision_fromtype(::Type{<:Reactant.RNumber{T}}) where T = get_precision_fromtype(T)

function allocate_array(cunit::ReactantUnit, ::Type{T}, dims::Dims) where T
    U = Reactant.unwrapped_eltype(T)
    if within_compile()
        return similar(TracedRArray{U}, dims)
    else
        dev = get_device(cunit)
        client = dev.client === missing ? nothing : dev.client
        device = dev.device === missing ? nothing : dev.device
        return ConcreteRArray{U}(undef, dims; client, device)
    end
end


function HeterogeneousComputing.on_device(f, device::ReactantDevice, dummy_args::Vararg{Any,N}) where {N}
    f_device, args_device = adapt(device, (f, dummy_args))
    f_compiled = compile(f_device, args_device)
    # Reactant-compiled functions are probably not thread-safe:
    lock = ReentrantLock()
    return _OnDevice{N}(f_compiled, device, lock)
end

end # module HeterogeneousComputingReactantExt
