# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

# Threefry-4x64 with 20 rounds, as specified by Random123 (Salmon et al.,
# "Parallel random numbers: as easy as 1, 2, 3", SC 2011). For a fixed key,
# the map from counter to output is a bijection. Used here as a pseudorandom
# function to derive stream keys, not as a random number generator.

const _THREEFRY_PARITY = 0x1bd11bdaa9fc1a22

# Rotation constants of Threefry-4x64, two per round. Rounds cycle through
# them with period 8, so four consecutive rounds use either half:
const _THREEFRY_ROTATIONS_A = ((14, 16), (52, 57), (23, 40), (5, 37))
const _THREEFRY_ROTATIONS_B = ((25, 33), (46, 12), (58, 22), (32, 32))

@inline function _threefry_mix(x0::UInt64, x1::UInt64, rot::Int)
    x0 += x1
    x1 = bitrotate(x1, rot) ⊻ x0
    return x0, x1
end

# Four rounds, alternating between the two word pairings:
@inline function _threefry_rounds(x::NTuple{4,UInt64}, rot::NTuple{4,NTuple{2,Int}})
    x0, x1, x2, x3 = x
    x0, x1 = _threefry_mix(x0, x1, rot[1][1])
    x2, x3 = _threefry_mix(x2, x3, rot[1][2])
    x0, x3 = _threefry_mix(x0, x3, rot[2][1])
    x2, x1 = _threefry_mix(x2, x1, rot[2][2])
    x0, x1 = _threefry_mix(x0, x1, rot[3][1])
    x2, x3 = _threefry_mix(x2, x3, rot[3][2])
    x0, x3 = _threefry_mix(x0, x3, rot[4][1])
    x2, x1 = _threefry_mix(x2, x1, rot[4][2])
    return (x0, x1, x2, x3)
end

# Key injection number `s` after every fourth round:
@inline function _threefry_inject(x::NTuple{4,UInt64}, ks::NTuple{5,UInt64}, s::Int)
    return (
        x[1] + ks[s % 5 + 1],
        x[2] + ks[(s + 1) % 5 + 1],
        x[3] + ks[(s + 2) % 5 + 1],
        x[4] + ks[(s + 3) % 5 + 1] + UInt64(s)
    )
end

@inline function _threefry4x64(key::NTuple{4,UInt64}, ctr::NTuple{4,UInt64})
    k0, k1, k2, k3 = key
    ks = (k0, k1, k2, k3, _THREEFRY_PARITY ⊻ k0 ⊻ k1 ⊻ k2 ⊻ k3)
    x = (ctr[1] + k0, ctr[2] + k1, ctr[3] + k2, ctr[4] + k3)
    x = _threefry_inject(_threefry_rounds(x, _THREEFRY_ROTATIONS_A), ks, 1)
    x = _threefry_inject(_threefry_rounds(x, _THREEFRY_ROTATIONS_B), ks, 2)
    x = _threefry_inject(_threefry_rounds(x, _THREEFRY_ROTATIONS_A), ks, 3)
    x = _threefry_inject(_threefry_rounds(x, _THREEFRY_ROTATIONS_B), ks, 4)
    x = _threefry_inject(_threefry_rounds(x, _THREEFRY_ROTATIONS_A), ks, 5)
    return x
end
