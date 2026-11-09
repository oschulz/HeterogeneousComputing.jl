# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

using Random
using MLDataDevices: CPUDevice
using JLArrays
import GPUArrays
using Adapt: adapt

using HeterogeneousComputing: _nonzero_xoshiro_state, _symbol_digest, _seeded_rng_like


@testset "streamkey" begin
    @testset "roots" begin
        @test @inferred(rootrng(42)) isa KeyedRNG{Xoshiro}
        @test rootrng(42) == rootrng(CPUnit(), 42) == rootrng(CPUDevice(), 42)
        @test rootrng(42) != rootrng(43)
        @test rootrng(typemax(UInt64)) isa KeyedRNG{Xoshiro}
        @test_throws ArgumentError rootrng(-1)
        @test_throws ArgumentError rootrng(true)

        @test has_rng_streamkey(rootrng(42))
        @test !has_rng_streamkey(Xoshiro(1))
        @test !has_rng_streamkey(Random.default_rng())
        @test rng_streamkey(rootrng(42)) isa NTuple{4,UInt64}
        @test_throws ArgumentError rng_streamkey(Xoshiro(1))
        @test_throws ArgumentError subrng(Xoshiro(1), 1)

        # Roots from an entropy source consume eight 32-bit values from it:
        entropy = Xoshiro(7)
        r = rootrng(entropy)
        @test r == rootrng(Xoshiro(7))
        @test entropy == (e = Xoshiro(7); rand(e, UInt32, 8); e)
        @test r != rootrng(entropy)
        @test rootrng() isa KeyedRNG{Xoshiro}
        @test rootrng() != rootrng()

        # A root created from a key restarts that stream:
        @test rootrng(rng_streamkey(r)) == r
        c = subrng(r, :a)
        @test rootrng(rng_streamkey(c)) == c
    end

    @testset "ids" begin
        r = rootrng(42)
        @test subrng(r, 1) == subrng(r, UInt8(1)) == subrng(r, Int128(1))
        @test subrng(r, 1) != subrng(r, Symbol("1"))
        @test subrng(r, :a, :b) != subrng(r, :ab)
        @test subrng(r, 0) != subrng(r, 1)
        @test_throws ArgumentError subrng(r, -1)
        @test_throws ArgumentError subrng(r, true)
        @test_throws ArgumentError subrng(r, big(2)^64)
        @test_throws ArgumentError subrng(r, 1.5)
        @test_throws ArgumentError subrng(r, "a")
        @test_throws MethodError subrng(r)
        @test _symbol_digest(:a) isa NTuple{3,UInt64}
        @test _symbol_digest(:a) != _symbol_digest(:b)
    end

    @testset "derivation" begin
        r = rootrng(42)
        c1, c2 = subrng(r, :walker, 1), subrng(r, :walker, 2)
        @test @inferred(subrng(r, :walker, 1)) isa KeyedRNG{Xoshiro}
        @test @inferred(subrng(r, 7)) isa KeyedRNG{Xoshiro}
        @test c1 != c2
        @test rng_streamkey(c1) != rng_streamkey(c2)
        @test rng_streamkey(c1) != rng_streamkey(r)
        @test subrng(r, :walker, 1) == c1
        @test subrng(subrng(r, :walker), 1) == c1
        @test rng_streamkey(subrng(c1, 3)) == rng_streamkey(subrng(r, :walker, 1, 3))
        @test length(Set(rng_streamkey(subrng(r, i)) for i in 1:1000)) == 1000
        @test rand(subrng(r, 1), 10) != rand(subrng(r, 2), 10)

        # Deriving children doesn't advance the parent, and drawing from the
        # parent or from children doesn't change other children:
        r_before = copy(r)
        subrng(r, 5)
        @test r == r_before
        rand(r, 1000)
        rand(c1, 10)
        @test subrng(r, :walker, 1) == subrng(r_before, :walker, 1)
        @test subrng(r, :walker, 2) == c2
    end

    @testset "Random API" begin
        r = rootrng(42)
        leaf = copy(r).rng
        draws = (
            rng -> rand(rng), rng -> rand(rng, Float32), rng -> rand(rng, Float16), rng -> rand(rng, UInt64),
            rng -> rand(rng, Int32), rng -> rand(rng, Bool), rng -> rand(rng, UInt128), rng -> randn(rng),
            rng -> randn(rng, Float32), rng -> randexp(rng), rng -> rand(rng, 1:10), rng -> rand(rng, 'a':'z'),
            rng -> rand(rng, [1.5, 2.5]), rng -> rand(rng, 3), rng -> rand(rng, Int, 2, 2), rng -> randn(rng, 4),
            rng -> randexp(rng, Float32, 2), rng -> rand(rng, Float64, 10^4), rng -> randstring(rng),
            rng -> shuffle(rng, 1:10), rng -> randperm(rng, 10)
        )
        for draw in draws
            @test draw(copy(r)) == draw(copy(leaf))
        end
        for fill! in (rand!, randn!, randexp!)
            A, B = zeros(5), zeros(5)
            @test fill!(copy(r), A) == fill!(copy(leaf), B)
        end
        @test rand!(copy(r), zeros(Int, 5), 1:5) == rand!(copy(leaf), zeros(Int, 5), 1:5)
        @test rand!(copy(r), zeros(3), Float32) == rand!(copy(leaf), zeros(3), Float32)

        # Drawing advances the wrapped RNG, reseeding keeps the key:
        a = copy(r)
        rand(a)
        @test a != r && a.rng != r.rng
        Random.seed!(a, 1)
        @test rng_streamkey(a) == rng_streamkey(r) && a.rng == Xoshiro(1)
        Random.seed!(a)
        @test rng_streamkey(a) == rng_streamkey(r) && a.rng != Xoshiro(1)

        @test copy(r) == r && copy(r) !== r && copy(r).rng !== r.rng
        @test deepcopy(r) == r
        @test hash(copy(r)) == hash(r)
        @test hash(r) != hash(subrng(r, 1))
        @test occursin("streamkey", repr(r)) && occursin("Xoshiro", repr(r))

        z = ntuple(_ -> UInt64(0), 4)
        @test _nonzero_xoshiro_state(z) == (UInt64(1), UInt64(0), UInt64(0), UInt64(0))
        w = (UInt64(5), UInt64(0), UInt64(0), UInt64(0))
        @test _nonzero_xoshiro_state(w) === w
        @test _seeded_rng_like(Xoshiro(1), CPUnit(), z) == Xoshiro(1, 0, 0, 0)
        @test _seeded_rng_like(Random.default_rng(), CPUnit(), w) == Xoshiro(5, 0, 0, 0)
    end

    @testset "contexts" begin
        ctx = rootcontext(42)
        @test ctx isa GenContext{Float64,CPUnit,KeyedRNG{Xoshiro}}
        @test get_rng(ctx) == rootrng(42)
        @test get_rng(rootcontext(CPUnit(), 42)) == rootrng(42)
        @test get_rng(rootcontext(CPUDevice(), 42)) == rootrng(42)
        @test get_rng(rootcontext(Xoshiro(7))) == rootrng(Xoshiro(7))
        @test get_rng(rootcontext()) isa KeyedRNG{Xoshiro}
        @test has_rng_streamkey(get_rng(ctx))
        @test GenContext{Float32}(get_rng(ctx)) isa GenContext{Float32,CPUnit,KeyedRNG{Xoshiro}}
        @test get_compute_unit(get_rng(ctx)) === ComputeUnitIndependent()

        sub = @inferred(subcontext(ctx, :chain, 3))
        @test sub isa GenContext{Float64,CPUnit,KeyedRNG{Xoshiro}}
        @test get_rng(sub) == subrng(get_rng(ctx), :chain, 3)
        @test get_precision(subcontext(GenContext{Float32}(CPUnit(), rootrng(1)), 1)) === Float32
        @test_throws ArgumentError subcontext(GenContext(), 1)
        @test adapt(CPUnit(), ctx) === ctx

        @test rand(rootcontext(42), 3) == rand(rootcontext(42), 3) == rand(rootrng(42), 3)
        test_gencontext_draws(ctx)
        test_gencontext_draws(GenContext{Float32}(CPUnit(), subrng(rootrng(1), :x)))
    end

    @testset "JLArrays" begin
        jl_unit = AbstractComputeUnit(JLBackend())
        ctx = rootcontext(42)

        # Moving a context keeps the key and derives a leaf of the target's kind:
        jl_ctx = adapt(jl_unit, ctx)
        @test get_rng(jl_ctx) isa KeyedRNG{<:GPUArrays.RNG{JLArray}}
        @test rng_streamkey(get_rng(jl_ctx)) == rng_streamkey(get_rng(ctx))
        @test get_compute_unit(get_rng(jl_ctx)) == jl_unit
        back = adapt(CPUnit(), subcontext(jl_ctx, 3))
        @test get_rng(back) isa KeyedRNG{Xoshiro}
        @test rng_streamkey(get_rng(back)) == rng_streamkey(get_rng(subcontext(ctx, 3)))

        # Keys don't depend on placement, children stay on their unit:
        @test rng_streamkey(get_rng(subcontext(jl_ctx, 3))) == rng_streamkey(get_rng(subcontext(ctx, 3)))
        @test get_rng(subcontext(jl_ctx, 3)) isa KeyedRNG{<:GPUArrays.RNG{JLArray}}
        @test get_compute_unit(subcontext(jl_ctx, 3)) == jl_unit

        jl_root = rootrng(jl_unit, 42)
        @test jl_root isa KeyedRNG{<:GPUArrays.RNG{JLArray}}
        @test rng_streamkey(jl_root) == rng_streamkey(rootrng(42))
        @test rootrng(jl_root) isa KeyedRNG{<:GPUArrays.RNG{JLArray}}
        @test rootrng(jl_unit, jl_root.rng) isa KeyedRNG{<:GPUArrays.RNG{JLArray}}
        @test Array(rand(rootcontext(jl_unit, 42), 5)) == Array(rand(rootcontext(jl_unit, 42), 5))
        @test Array(rand(subcontext(rootcontext(jl_unit, 42), 1), 5)) !=
              Array(rand(subcontext(rootcontext(jl_unit, 42), 2), 5))
        test_gencontext_draws(subcontext(rootcontext(jl_unit, 42), 1))
        test_gencontext_draws(GenContext{Float32}(jl_unit, subrng(jl_root, :x)))
    end
end
