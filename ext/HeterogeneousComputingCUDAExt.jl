# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingCUDAExt

import CUDA

using HeterogeneousComputing
import HeterogeneousComputing: ka_backend

import Adapt


"""
    struct CUDAUnit <: AbstractGPUnit

A CUDA-compatible GPU compute unit.

Constructors

```julia
CUDAUnit(device_number::Integer)
CUDAUnit(dev::CUDA.CUDA.CuDevice)
```
"""
struct CUDAUnit <: AbstractGPUnit
    devhandle::Int
end


CUDAUnit(dev::CUDA.CuDevice) = CUDAUnit(dev.handle)


HeterogeneousComputing.AbstractComputeSystem(dev::CUDA.CuDevice) = CUDAUnit(dev.handle)
Base.convert(::Type{AbstractComputeSystem}, dev::CUDA.CuDevice) = CUDAUnit(dev)

CUDA.CuDevice(cunit::CUDAUnit) = CUDA.CuDevice(cunit.devhandle)
Base.convert(::Type{CUDA.CuDevice}, cunit::CUDAUnit) = CUDA.CuDevice(cunit)

function HeterogeneousComputing.get_compute_unit_impl(@nospecialize(TypeHistory::Type), A::CUDA.CuArray)
    return CUDAUnit(CUDA.device(A))
end
function HeterogeneousComputing.get_compute_unit_impl(
    @nospecialize(TypeHistory::Type),
    A::CUDA.CUDA.CUSPARSE.AbstractCuSparseArray
)
    return CUDAUnit(CUDA.device(A.nzVal))
end


HeterogeneousComputing.get_total_memory(cunit::CUDAUnit) = CUDA.totalmem(CUDA.CuDevice(cunit))

function HeterogeneousComputing.get_free_memory(cunit::CUDAUnit)
    return unsigned(CUDA.device!(CUDA.available_memory, CUDA.CuDevice(cunit)))
end


function Adapt.adapt_storage(cunit::CUDAUnit, x)
    oldhandle = CUDA.device().handle
    try
        oldhandle != cunit.devhandle && CUDA.device!(cunit.devhandle)
        Adapt.adapt(CUDA.CuArray, x)
    finally
        oldhandle != cunit.devhandle && CUDA.device!(oldhandle)
    end
end


function HeterogeneousComputing.allocate_array(cunit::CUDAUnit, ::Type{T}, dims::Dims) where T
    oldhandle = CUDA.device().handle
    try
        oldhandle != cunit.devhandle && CUDA.device!(cunit.devhandle)
        CUDA.CuArray{T}(undef, dims)
    finally
        oldhandle != cunit.devhandle && CUDA.device!(oldhandle)
    end
end



# Requires KernelAbstractions v0.9 and CUDA v4:
@static if isdefined(CUDA, :CUDABackend)
    ka_backend(::CUDAUnit) = CUDA.CUDABackend()
    CUDA.CUDABackend(cunit::CUDAUnit) = ka_backend(cunit)
    Base.convert(::Type{CUDA.CUDABackend}, cunit::CUDAUnit) = ka_backend(cunit)
end

end # module HeterogeneousComputingCUDAExt
