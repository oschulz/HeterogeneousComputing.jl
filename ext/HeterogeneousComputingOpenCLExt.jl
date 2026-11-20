# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingOpenCLExt

using HeterogeneousComputing
import HeterogeneousComputing: _canonical_device, _within_unit, get_compute_unit_impl, allocate_array
import Adapt
using MLDataDevices: MLDataDevices, OpenCLDevice
using OpenCL: OpenCL, CLArray, cl

# OpenCL arrays belong to a context. Distinct contexts can use the same device.
struct OpenCLContextDevice <: MLDataDevices.AbstractGPUDevice
    context::cl.Context
end
const OpenCLUnit = DeviceUnit{OpenCLContextDevice}

Base.eltype(::OpenCLContextDevice) = nothing
Base.:(==)(a::OpenCLContextDevice, b::OpenCLContextDevice) = a.context == b.context
Base.hash(dev::OpenCLContextDevice, h::UInt) = hash(dev.context, h)
MLDataDevices.loaded(::Union{OpenCLContextDevice,Type{OpenCLContextDevice}}) = true
function MLDataDevices.functional(::Union{OpenCLContextDevice,Type{OpenCLContextDevice}})
    return MLDataDevices.functional(OpenCLDevice())
end

_canonical_device(::OpenCLDevice) = OpenCLContextDevice(cl.context())
get_compute_unit_impl(A::CLArray) = DeviceUnit(OpenCLContextDevice(OpenCL.context(A)))
_within_unit(f, cunit::OpenCLUnit) = _within_context(f, cunit.device.context)

function _within_context(f::F, ctx::cl.Context) where F
    previous_ctx, previous_queue = cl.context(), cl.queue()
    try
        cl.context!(ctx)
        if cl.queue().context != ctx
            # OpenCL caches a task's queue by device. A second context on
            # that device needs its own queue for successive commands.
            queues = get!(WeakKeyDict{cl.Context,cl.CmdQueue}, task_local_storage(), OpenCLContextDevice)
            cl.queue!(get!(cl.CmdQueue, queues, ctx))
        end
        return f()
    finally
        cl.context!(previous_ctx)
        cl.queue!(previous_queue)
    end
end

function MLDataDevices.default_device_rng(dev::OpenCLContextDevice)
    return _within_context(() -> MLDataDevices.default_device_rng(OpenCLDevice()), dev.context)
end
MLDataDevices.default_device_rng(cunit::OpenCLUnit) = MLDataDevices.default_device_rng(cunit.device)
function allocate_array(cunit::OpenCLUnit, ::Type{T}, dims::Dims) where T
    return _within_unit(() -> CLArray{T}(undef, dims), cunit)
end

function Adapt.adapt_storage(dev::OpenCLContextDevice, A::AbstractArray)
    return _within_context(() -> Adapt.adapt_storage(OpenCLDevice{Nothing}(), A), dev.context)
end
Adapt.adapt_storage(cunit::OpenCLUnit, A::AbstractArray) = Adapt.adapt_storage(cunit.device, A)
function Adapt.adapt_storage(dev::OpenCLContextDevice, A::CLArray)
    source = OpenCL.context(A)
    source == dev.context && return A
    # OpenCL buffers in different contexts cannot be copied directly.
    host = _within_context(() -> Array(A), source)
    return Adapt.adapt_storage(dev, host)
end
Adapt.adapt_storage(::CPUnit, A::CLArray) = _within_context(() -> Array(A), OpenCL.context(A))

end # module HeterogeneousComputingOpenCLExt
