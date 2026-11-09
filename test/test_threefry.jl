# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

using Random
using Random123: threefry

using HeterogeneousComputing: _threefry4x64


@testset "Threefry-4x64" begin
    zeros4 = ntuple(_ -> UInt64(0), 4)
    ones4 = ntuple(_ -> typemax(UInt64), 4)
    # Known-answer vectors of Random123 for Threefry-4x64 with 20 rounds:
    @test _threefry4x64(zeros4, zeros4) ==
          (0x09218ebde6c85537, 0x55941f5266d86105, 0x4bd25e16282434dc, 0xee29ec846bd2e40b)
    @test _threefry4x64(ones4, ones4) ==
          (0x29c24097942bba1b, 0x0371bbfb0f6f4e11, 0x3c231ffa33f83a1c, 0xcd29113fde32d168)

    rng = Xoshiro(0x7f3e)
    randwords() = ntuple(_ -> rand(rng, UInt64), Val(4))
    @test all(1:1000) do _
        key, ctr = randwords(), randwords()
        _threefry4x64(key, ctr) == threefry(key, ctr, Val(20))
    end

    # Every input bit matters:
    key, ctr = randwords(), randwords()
    ref = _threefry4x64(key, ctr)
    flip(t, i, b) = ntuple(j -> j == i ? t[j] ⊻ (UInt64(1) << b) : t[j], Val(4))
    @test all(_threefry4x64(flip(key, i, b), ctr) != ref for i in 1:4, b in (0, 31, 63))
    @test all(_threefry4x64(key, flip(ctr, i, b)) != ref for i in 1:4, b in (0, 31, 63))
    @test @inferred(_threefry4x64(key, ctr)) isa NTuple{4,UInt64}
    @test @allocated(_threefry4x64(key, ctr)) == 0
end
