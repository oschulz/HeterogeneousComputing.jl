# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingGPUArraysExt

import GPUArrays

import HeterogeneousComputing: get_compute_unit_impl, _rng_cunit

using MLDataDevices: get_device


# GPUArrays.RNG isn't tied to a specific device, its compute unit is the
# currently active device of its array type:
get_compute_unit_impl(::GPUArrays.RNG{AT}) where AT = _rng_cunit(get_device(AT))

end # module HeterogeneousComputingGPUArraysExt
