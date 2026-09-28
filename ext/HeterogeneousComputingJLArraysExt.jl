# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingJLArraysExt

using JLArrays: JLArray, JLBackend
import GPUArrays

using HeterogeneousComputing
import HeterogeneousComputing: get_total_memory, get_free_memory, get_compute_unit_impl

import Adapt
import MLDataDevices


"""
    struct JLArraysDevice <: MLDataDevices.AbstractDevice

Device for JLArrays, a reference implementation of GPU arrays that runs on the
CPU. Useful for testing GPU code paths without a GPU.

Use `AbstractComputeUnit(JLArrays.JLBackend())` to get a compute unit based on
a `JLArraysDevice`.
"""
struct JLArraysDevice <: MLDataDevices.AbstractDevice end

const JLArraysUnit = DeviceUnit{JLArraysDevice}

MLDataDevices.loaded(::Union{JLArraysDevice,Type{<:JLArraysDevice}}) = true
MLDataDevices.functional(::Union{JLArraysDevice,Type{<:JLArraysDevice}}) = true

Adapt.adapt_storage(::JLArraysDevice, x::AbstractArray) = Adapt.adapt(JLArray, x)
MLDataDevices.default_device_rng(::JLArraysDevice) = GPUArrays.RNG{JLArray}()

HeterogeneousComputing.AbstractComputeUnit(::JLBackend) = DeviceUnit(JLArraysDevice())

get_compute_unit_impl(::JLArray) = DeviceUnit(JLArraysDevice())
get_compute_unit_impl(::GPUArrays.RNG{JLArray}) = DeviceUnit(JLArraysDevice())

# MLDataDevices considers JLArrays to be CPU arrays:
Adapt.adapt_storage(::CPUnit, A::JLArray) = Array(A)

get_total_memory(::JLArraysUnit) = Sys.total_memory()
get_free_memory(::JLArraysUnit) = Sys.free_memory()

end # module HeterogeneousComputingJLArraysExt
