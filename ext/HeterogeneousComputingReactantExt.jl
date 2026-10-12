# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingReactantExt

using HeterogeneousComputing
using HeterogeneousComputing: _OnDevice
import HeterogeneousComputing: get_compute_unit_impl, get_precision_fromtype, allocate_array, compute_unit_mergerule
import HeterogeneousComputing:
    _canonical_device, _same_device, _fill_random!, _draw_scalar, _scalar_randfun, _RecurseFields, _gen_convert

import Reactant
using Reactant: ConcreteRArray, TracedRArray, TracedRNumber, ReactantRNG, within_compile
using Reactant: XLA, Sharding
using Reactant.Compiler: compile

import Adapt
using Adapt: adapt
using MLDataDevices: MLDataDevices, ReactantDevice, get_device, default_device_rng
import Random


const ReactantUnit = DeviceUnit{<:ReactantDevice}

const _ConcreteTypes = Union{
    Reactant.ConcretePJRTArray,Reactant.ConcretePJRTNumber,
    Reactant.ConcreteIFRTArray,Reactant.ConcreteIFRTNumber
}

# A unit is a client and a device, or a client and the mesh of devices that
# sharded arrays are distributed over. The unit keeps the mesh, but not how
# individual arrays are partitioned over it. New arrays on the unit are
# replicated over the mesh, like MLDataDevices does. Outside of compiled code,
# a missing client or device is resolved like Reactant does when moving data,
# within compiled code the location of values is unknown:
function _canonical_device(dev::ReactantDevice)
    mesh_sharding = _mesh_sharding(dev.sharding)
    if within_compile()
        client, device = dev.client, dev.device
    elseif mesh_sharding === missing
        client, device = _resolve_location(dev.client, something(dev.device, missing))
    else
        client, device = coalesce(dev.client, XLA.default_backend()), nothing
    end
    return MLDataDevices.with_eltype(ReactantDevice(client, device, mesh_sharding), nothing)
end

_resolve_location(client, device) = (client, device)
_resolve_location(::Missing, ::Missing) = _resolve_location(XLA.default_backend(), missing)
_resolve_location(client, ::Missing) = (client, XLA.default_device(client))
_resolve_location(::Missing, device::XLA.AbstractDevice) = (XLA.client(device), device)

_mesh_sharding(::Missing) = missing
_mesh_sharding(::Sharding.NoSharding) = missing
_mesh_sharding(sharding::Sharding.AbstractSharding) = Sharding.Replicated(sharding.mesh)

# MLDataDevices records the sharding of each array separately:
function _mesh_sharding(shardings::IdDict)
    mesh_shardings = filter(!ismissing, map(_mesh_sharding, collect(values(shardings))))
    isempty(mesh_shardings) && return missing
    allequal(s.mesh.device_ids for s in mesh_shardings) ||
        throw(ArgumentError("Arrays are sharded over different device meshes"))
    return first(mesh_shardings)
end

get_compute_unit_impl(x::_ConcreteTypes) = DeviceUnit(get_device(x))
# The device of traced values is unknown within compiled code:
function get_compute_unit_impl(::Union{TracedRArray,TracedRNumber})
    return DeviceUnit(ReactantDevice())
end
# get_device fails for ReactantRNGs within compiled code, use their fields:
get_compute_unit_impl(::ReactantRNG) = _RecurseFields()

# Units derived within compiled code specify neither client nor device, and
# merge with any unit that does:
_is_partial(dev::ReactantDevice) = dev.client === missing

# MLDataDevices treats unspecified and sharded (device `nothing`) locations as
# equal to any location, units need a strict equality:
function _same_device(a::ReactantDevice, b::ReactantDevice)
    return _location_id(a.client) === _location_id(b.client) &&
           _location_id(a.device) === _location_id(b.device) &&
           _same_mesh(a.sharding, b.sharding)
end

_location_id(x::Union{Missing,Nothing}) = x
_location_id(client::XLA.AbstractClient) = client.client
_location_id(device::XLA.AbstractDevice) = device.device

_same_mesh(::Missing, ::Missing) = true
_same_mesh(a::Sharding.Replicated, b::Sharding.Replicated) = a.mesh.device_ids == b.mesh.device_ids
_same_mesh(::Any, ::Any) = false

function compute_unit_mergerule(a::ReactantUnit, b::ReactantUnit)
    return _is_partial(a.device) && !_is_partial(b.device) ? b : HeterogeneousComputing.NoCUnitMergeRule()
end

# A new RNG created within compiled code would have a fixed seed:
function MLDataDevices.default_device_rng(cunit::ReactantUnit)
    within_compile() && throw(ArgumentError("Random number generators for Reactant must be passed into compiled code"))
    # MLDataDevices.default_device_rng(::ReactantDevice) ignores client and device:
    return adapt(get_device(cunit), Random.RandomDevice())
end

# MLDataDevices doesn't move ReactantRNGs between devices:
function Adapt.adapt_storage(cunit::ReactantUnit, rng::ReactantRNG)
    return get_compute_unit(rng) == cunit ? rng : ReactantRNG(adapt(get_device(cunit), rng.seed), rng.algorithm)
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

_gen_convert(::Type{T}, x::TracedRNumber{<:AbstractFloat}) where T = convert(TracedRNumber{T}, x)

function allocate_array(cunit::ReactantUnit, ::Type{T}, dims::Dims) where T
    U = Reactant.unwrapped_eltype(T)
    if within_compile()
        return similar(TracedRArray{U}, dims)
    else
        dev = get_device(cunit)
        sharding = coalesce(dev.sharding, Sharding.NoSharding())
        return ConcreteRArray{U}(undef, dims; client = dev.client, device = dev.device, sharding)
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
