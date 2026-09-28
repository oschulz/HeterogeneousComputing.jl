# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingGPUArraysExt

import GPUArrays

import HeterogeneousComputing: _fill_random!, _draw_scalar, _randexp_from_rand!

import Random


# GPUArrays.RNG supports neither exponential nor scalar draws:

_fill_random!(::typeof(Random.randexp!), rng::GPUArrays.RNG, A::AbstractArray) = _randexp_from_rand!(rng, A)

function _draw_scalar(f!::F, rng::GPUArrays.RNG{AT}, ::Type{T}) where {F,AT,T}
    return Array(_fill_random!(f!, rng, similar(AT{T}, 1)))[1]
end

end # module HeterogeneousComputingGPUArraysExt
