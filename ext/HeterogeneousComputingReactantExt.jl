# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingReactantExt

using HeterogeneousComputing
using HeterogeneousComputing: _OnDevice
import HeterogeneousComputing: get_compute_unit_impl, get_precision_fromtype, allocate_array, compute_unit_mergerule
import HeterogeneousComputing:
    _canonical_device, _same_device, _fill_random!, _draw_scalar, _scalar_randfun, _RecurseFields

import Reactant
using Reactant: ConcreteRArray, TracedRArray, TracedRNumber, ReactantRNG, within_compile
using Reactant.Compiler: compile

using Adapt: adapt
using MLDataDevices: MLDataDevices, ReactantDevice, get_device, default_device_rng


const ReactantUnit = DeviceUnit{<:ReactantDevice}

const _ConcreteTypes = Union{
    Reactant.ConcretePJRTArray,Reactant.ConcretePJRTNumber,
    Reactant.ConcreteIFRTArray,Reactant.ConcreteIFRTNumber
}

# Units don't track the sharding of individual arrays:
function _canonical_device(dev::ReactantDevice)
    return MLDataDevices.with_eltype(ReactantDevice(dev.client, dev.device, missing), nothing)
end

get_compute_unit_impl(x::_ConcreteTypes) = DeviceUnit(get_device(x))
# The device of traced values is unknown within compiled code:
function get_compute_unit_impl(::Union{TracedRArray,TracedRNumber})
    return DeviceUnit(ReactantDevice())
end
# get_device fails for ReactantRNGs within compiled code, use their fields:
get_compute_unit_impl(::ReactantRNG) = _RecurseFields()

# Devices derived within compiled code specify neither client nor device, and
# merge with any device that does:
_is_partial(dev::ReactantDevice) = dev.client === missing

_same_device(a::ReactantDevice, b::ReactantDevice) = _is_partial(a) == _is_partial(b) && a == b

function compute_unit_mergerule(a::ReactantUnit, b::ReactantUnit)
    return _is_partial(a.device) && !_is_partial(b.device) ? b : HeterogeneousComputing.NoCUnitMergeRule()
end

# A new RNG created within compiled code would have a fixed seed:
function MLDataDevices.default_device_rng(cunit::ReactantUnit)
    within_compile() && throw(ArgumentError("Random number generators for Reactant must be passed into compiled code"))
    return default_device_rng(get_device(cunit))
end

const _ConcreteRNG = ReactantRNG{<:Reactant.AbstractConcreteArray}

function _no_eager_draws()
    return throw(
        ArgumentError("Random numbers from Reactant random number generators can only be drawn within compiled code")
    )
end

_fill_random!(::F, ::_ConcreteRNG, ::AbstractArray) where {F} = _no_eager_draws()
_draw_scalar(::F, ::_ConcreteRNG, ::Type{T}, ::AbstractComputeUnit) where {F,T} = _no_eager_draws()
# Reactant supports scalar draws within compiled code:
function _draw_scalar(f!::F, rng::ReactantRNG{<:TracedRArray}, ::Type{T}, ::AbstractComputeUnit) where {F,T}
    return _scalar_randfun(f!)(rng, T)
end

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
