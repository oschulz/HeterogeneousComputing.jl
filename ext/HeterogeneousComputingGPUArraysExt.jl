# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingGPUArraysExt

import GPUArrays

using HeterogeneousComputing: AbstractComputeUnit
import HeterogeneousComputing: get_compute_unit_impl, _rng_cunit, _seeded_rng_like

using MLDataDevices: get_device


# GPUArrays.RNG isn't tied to a specific device, its compute unit is the
# currently active device of its array type:
get_compute_unit_impl(::GPUArrays.RNG{AT}) where AT = _rng_cunit(get_device(AT))

_seeded_rng_like(rng::GPUArrays.RNG, ::AbstractComputeUnit, seeds::NTuple{4,UInt64}) = typeof(rng)(seeds[1])

end # module HeterogeneousComputingGPUArraysExt
