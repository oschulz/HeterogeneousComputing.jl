# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingGPUArraysExt

import GPUArrays

using HeterogeneousComputing: AbstractComputeUnit
import HeterogeneousComputing: get_compute_unit_impl, _rng_cunit, _seeded_rng_like
import HeterogeneousComputing: _copy_rng, _isequal_rng, _hash_rng

using MLDataDevices: get_device


# GPUArrays.RNG isn't tied to a specific device, its compute unit is the
# currently active device of its array type:
get_compute_unit_impl(::GPUArrays.RNG{AT}) where AT = _rng_cunit(get_device(AT))

_seeded_rng_like(rng::GPUArrays.RNG, ::AbstractComputeUnit, seeds::NTuple{4,UInt64}) = typeof(rng)(seeds[1])

# GPUArrays.RNG supports neither `copy` nor `==`:
_copy_rng(rng::GPUArrays.RNG) = typeof(rng)(rng.seed, rng.counter)
function _isequal_rng(a::GPUArrays.RNG, b::GPUArrays.RNG)
    return typeof(a) == typeof(b) && a.seed == b.seed && a.counter == b.counter
end
_hash_rng(rng::GPUArrays.RNG, h::UInt) = hash(rng.counter, hash(rng.seed, hash(typeof(rng), h)))

end # module HeterogeneousComputingGPUArraysExt
