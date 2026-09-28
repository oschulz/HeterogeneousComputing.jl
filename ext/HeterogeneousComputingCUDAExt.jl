# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingCUDAExt

import CUDA

using HeterogeneousComputing
import HeterogeneousComputing: ka_backend, allocate_array, get_total_memory, get_free_memory
import HeterogeneousComputing: get_compute_unit_impl, _device_loaded, _canonical_device, _within_unit

using MLDataDevices: CUDADevice


const CUDAUnit = DeviceUnit{<:CUDADevice}

# MLDataDevices requires cuDNN to consider CUDA loaded, HeterogeneousComputing
# doesn't:
_device_loaded(::CUDADevice) = true

_canonical_device(::CUDADevice{Nothing}) = _canonical_device(CUDADevice(CUDA.device()))

HeterogeneousComputing.AbstractComputeUnit(dev::CUDA.CuDevice) = DeviceUnit(CUDADevice(dev))
Base.convert(::Type{AbstractComputeUnit}, dev::CUDA.CuDevice) = AbstractComputeUnit(dev)

_cudevice(cunit::CUDAUnit) = cunit.device.device


get_total_memory(cunit::CUDAUnit) = CUDA.totalmem(_cudevice(cunit))

function get_free_memory(cunit::CUDAUnit)
    @static if isdefined(CUDA, :free_memory)
        return unsigned(CUDA.device!(CUDA.free_memory, _cudevice(cunit)))
    else
        return unsigned(CUDA.device!(CUDA.available_memory, _cudevice(cunit)))
    end
end

_within_unit(f, cunit::CUDAUnit) = CUDA.device!(f, _cudevice(cunit))

allocate_array(cunit::CUDAUnit, ::Type{T}, dims::Dims) where T = _within_unit(() -> CUDA.CuArray{T}(undef, dims), cunit)

ka_backend(::CUDAUnit) = CUDA.CUDABackend()


# MLDataDevices doesn't know the cuRAND RNGs of CUDA v6:
@static if isdefined(CUDA, :cuRAND)
    get_compute_unit_impl(::Union{CUDA.cuRAND.LibraryRNG,CUDA.cuRAND.NativeRNG}) = DeviceUnit(CUDADevice(CUDA.device()))
end

end # module HeterogeneousComputingCUDAExt
