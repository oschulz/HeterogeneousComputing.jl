# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingCUDAExt

import CUDA

using HeterogeneousComputing
import HeterogeneousComputing: ka_backend, allocate_array, get_total_memory, get_free_memory
import HeterogeneousComputing: _canonical_device, _within_unit, _fill_random!, _draw_scalar, _randexp_from_rand!

import Random
using MLDataDevices: CUDADevice


const CUDAUnit = DeviceUnit{<:CUDADevice}

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


# CUDA v5 has its own RNG type (CUDA v6 uses GPUArrays.RNG), with no
# exponential or scalar draws:
@static if pkgversion(CUDA) < v"6"
    _fill_random!(::typeof(Random.randexp!), rng::CUDA.RNG, A::AbstractArray) = _randexp_from_rand!(rng, A)

    function _draw_scalar(f!::F, rng::CUDA.RNG, ::Type{T}) where {F,T}
        return Array(_fill_random!(f!, rng, CUDA.CuArray{T}(undef, 1)))[1]
    end
end

end # module HeterogeneousComputingCUDAExt
