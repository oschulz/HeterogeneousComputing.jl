# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingJLArraysExt

using JLArrays: JLArray, JLBackend
import GPUArrays

using HeterogeneousComputing
import HeterogeneousComputing: ka_backend, allocate_array, get_total_memory, get_free_memory
import HeterogeneousComputing: get_compute_unit_impl

import Adapt
import MLDataDevices


"""
    struct JLArraysUnit <: AbstractComputeUnit

Compute unit for JLArrays, a reference implementation of GPU arrays that runs
on the CPU. Useful for testing GPU code paths without a GPU.

Use `AbstractComputeUnit(JLArrays.JLBackend())` to get a `JLArraysUnit`.
"""
struct JLArraysUnit <: AbstractComputeUnit end

HeterogeneousComputing.AbstractComputeUnit(::JLBackend) = JLArraysUnit()

get_compute_unit_impl(@nospecialize(TypeHistory::Type), ::JLArray) = JLArraysUnit()
get_compute_unit_impl(@nospecialize(TypeHistory::Type), ::GPUArrays.RNG{JLArray}) = JLArraysUnit()

Adapt.adapt_storage(::JLArraysUnit, x) = Adapt.adapt_storage(JLArray, x)
# MLDataDevices considers JLArrays to be CPU arrays:
Adapt.adapt_storage(::CPUnit, A::JLArray) = Array(A)

MLDataDevices.default_device_rng(::JLArraysUnit) = GPUArrays.RNG{JLArray}()

allocate_array(::JLArraysUnit, ::Type{T}, dims::Dims) where T = JLArray{T}(undef, dims)

get_total_memory(::JLArraysUnit) = Sys.total_memory()
get_free_memory(::JLArraysUnit) = Sys.free_memory()

ka_backend(::JLArraysUnit) = JLBackend()

end # module HeterogeneousComputingJLArraysExt
