# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

using Random

@testset "rng" begin
    rng = Random.default_rng()

    @test @inferred(get_rng(42)) === HeterogeneousComputing.NoRNG{Int}()
    @test @inferred(get_rng(rng)) === rng
end
