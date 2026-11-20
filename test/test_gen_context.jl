# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

using Random
using MLDataDevices: CPUDevice, cpu_device
import MLDataDevices
using JLArrays
import GPUArrays
using Adapt: adapt

struct _AlternateHost <: AbstractComputeUnit end
HeterogeneousComputing.is_host_unit(::_AlternateHost) = true
MLDataDevices.default_device_rng(::_AlternateHost) = MersenneTwister(1)

struct _UnseedableUnit <: AbstractComputeUnit end
MLDataDevices.default_device_rng(::_UnseedableUnit) = RandomDevice()

struct _RejectingUnit <: AbstractComputeUnit end
function HeterogeneousComputing._seeded_rng(::_RejectingUnit, seeds::NTuple{4,UInt64})
    all(iszero, seeds) || throw(DomainError(seeds))
    return Xoshiro(1)
end

@testset "gen_context" begin
    RNG = typeof(Random.default_rng())

    @test @inferred(GenContext()) isa GenContext{Float64,CPUnit,RNG}
    @test @inferred(GenContext(CPUnit())) isa GenContext{Float64,CPUnit,RNG}
    @test @inferred(GenContext(Random.default_rng())) isa GenContext{Float64,CPUnit,RNG}
    @test @inferred(GenContext(CPUnit(), Random.default_rng())) isa GenContext{Float64,CPUnit,RNG}
    @test @inferred(GenContext{Float32}()) isa GenContext{Float32,CPUnit,RNG}
    @test @inferred(GenContext{Float32}(CPUnit())) isa GenContext{Float32,CPUnit,RNG}
    @test @inferred(GenContext{Float32}(Random.default_rng())) isa GenContext{Float32,CPUnit,RNG}
    @test @inferred(GenContext{Float32}(CPUnit(), Random.default_rng())) isa GenContext{Float32,CPUnit,RNG}

    @test @inferred(GenContext(CPUDevice())) isa GenContext{Float64,CPUnit,RNG}
    @test @inferred(GenContext(cpu_device(Float32))) isa GenContext{Float32,CPUnit,RNG}
    @test @inferred(GenContext(cpu_device(Float32), Xoshiro(42))) isa GenContext{Float32,CPUnit,Xoshiro}
    @test @inferred(GenContext{Float16}(cpu_device(Float32))) isa GenContext{Float16,CPUnit,RNG}

    cpunit = CPUnit()
    rng = Random.default_rng()
    ctx = GenContext{Float32}()

    host_rng = MersenneTwister(2)
    @test get_rng(adapt(_AlternateHost(), GenContext(host_rng))) === host_rng
    src = Xoshiro(4)
    @test_throws ArgumentError adapt(_UnseedableUnit(), GenContext(src))
    @test_throws DomainError adapt(_RejectingUnit(), GenContext(src))
    @test rand(src, UInt64, 8) == rand(Xoshiro(4), UInt64, 8)

    @test @inferred(GenContext(ctx)) isa typeof(ctx)
    @test @inferred(GenContext{Float16}(ctx)) isa GenContext{Float16}
    @test @inferred(convert(GenContext{Float16}, ctx)) isa typeof(GenContext{Float16}())
    @test @inferred((typeof(ctx))(ctx)) isa typeof(ctx)

    @test @inferred(get_precision(ctx)) === Float32
    @test @inferred(get_compute_unit(ctx)) === cpunit
    @test @inferred(get_rng(ctx)) === rng

    @test @inferred(get_gencontext(ctx)) === ctx
    @test @inferred(get_gencontext(rand(Float32, 7))) isa GenContext{Float32,CPUnit,RNG}
    @test @inferred(get_gencontext(42)) === HeterogeneousComputing.NoGenContext{Int}()
    @test get_gencontext("foo") === HeterogeneousComputing.NoGenContext{String}()
    mixed_data = (rand(3), JLArray(rand(3)))
    @test get_gencontext(mixed_data) === HeterogeneousComputing.NoGenContext{typeof(mixed_data)}()
    @test_throws ArgumentError GenContext(MixedComputeSystem())
    @test_throws ArgumentError GenContext(_UnknownUnitRNG())
    @test adapt(CPUnit(), ctx) === ctx

    # Draws use the RNG of the context:
    @test rand(GenContext(Xoshiro(1)), 3) == rand(Xoshiro(1), 3)
    @test randn(GenContext{Float32}(Xoshiro(1))) == randn(Xoshiro(1), Float32)

    _check_array(A, ::Type{T}, sz::Dims{N}) where {T,N} = @test A isa AbstractArray{T,N} && size(A) == sz

    _check_array(@inferred(allocate_array(ctx, (4, 5))), Float32, (4, 5))
    _check_array(@inferred(allocate_array(ctx, 4, 5)), Float32, (4, 5))
    _check_array(@inferred(allocate_array(ctx, Float16, (4, 5))), Float16, (4, 5))
    _check_array(@inferred(allocate_array(ctx, Float16, 4, 5)), Float16, (4, 5))
    _check_array(@inferred(allocate_array(ctx)), Float32, ())
    @test @inferred(fill_array(ctx, 1.5, UInt8(2), big(3))) == fill(1.5f0, 2, 3)
    @test_throws InexactError fill_array(ctx, 1, big(typemax(Int)) + 1)
    @test @inferred(fill_array(ctx, true, 2)) == fill(true, 2)

    test_gencontext_draws(ctx)
    test_gencontext_draws(GenContext{Float64}(Xoshiro(123)))

    @testset "JLArrays" begin
        jl_unit = AbstractComputeUnit(JLBackend())
        jl_ctx = GenContext{Float32}(jl_unit)
        @test jl_ctx isa GenContext{Float32,typeof(jl_unit)}
        @test GenContext{Float32}(get_rng(jl_ctx)) === jl_ctx
        @test get_gencontext(allocate_array(jl_ctx, 3)) isa GenContext{Float32,typeof(jl_unit)}
        test_gencontext_draws(jl_ctx)

        # Exponential variates on wrapped GPU arrays:
        A = allocate_array(jl_ctx, 6)
        @test randexp!(jl_ctx, view(A, 1:2:5)) isa SubArray
        @test all(x -> 0 <= x < Inf, Array(A)[1:2:5])

        seeded() = GenContext{Float32}(jl_unit, rootrng(jl_unit, 8))
        for (draw, draw!) in ((rand, rand!), (randn, randn!), (randexp, randexp!))
            parent = JLArray(fill(-1.0f0, 8, 8))
            dest = PermutedDimsArray(view(parent, 1:2:7, 1:2:7), (2, 1))
            expected = fill(-1.0f0, 8, 8)
            expected[1:2:7, 1:2:7] .= permutedims(Array(draw(seeded(), size(dest))))
            @test draw!(seeded(), dest) === dest
            @test Array(parent) == expected
        end

        # A rounded unit exponential has P(X = 0) ≈ 2^-25 and P(X > 8) ≈ exp(-8).
        for rng in (rootrng(jl_unit, 19), Xoshiro(19))
            E = Array(randexp(GenContext{Float16}(jl_unit, rng), 2^16))
            @test count(iszero, E) <= 2
            @test any(>(8), E)
        end

        rng = rootrng(jl_unit, 42)
        initial_rng = copy(rng)
        flags = JLArray(fill(true, 8))
        @test_throws ArgumentError randexp!(GenContext(jl_unit, rng), flags)
        @test all(Array(flags)) && rng == initial_rng

        # Host RNGs stay with contexts moved to host units, otherwise a new
        # RNG native to the target unit is derived, reproducibly, and the
        # original RNG advances:
        host_rng = Xoshiro(1)
        @test adapt(CPUnit(), GenContext{Float32}(jl_unit, host_rng)).rng === host_rng
        derived_jl() = adapt(jl_unit, GenContext{Float32}(Xoshiro(1)))
        @test derived_jl() isa GenContext{Float32,typeof(jl_unit),<:GPUArrays.RNG}
        @test Array(rand(derived_jl(), 3)) == Array(rand(derived_jl(), 3))
        adapt(jl_unit, GenContext{Float32}(host_rng))
        @test rand(host_rng) != rand(Xoshiro(1))
        derived_cpu() = adapt(CPUnit(), GenContext{Float32}(jl_unit, GPUArrays.RNG{JLArray}(42)))
        @test derived_cpu() isa GenContext{Float32,CPUnit,Xoshiro}
        @test rand(derived_cpu(), 3) == rand(derived_cpu(), 3)
        @test rand(adapt(CPUnit(), jl_ctx), 3) != rand(adapt(CPUnit(), jl_ctx), 3)
        Random.seed!(7)
        from_default_1 = adapt(jl_unit, GenContext{Float32}())
        Random.seed!(7)
        from_default_2 = adapt(jl_unit, GenContext{Float32}())
        @test from_default_1 isa GenContext{Float32,typeof(jl_unit),<:GPUArrays.RNG}
        @test Array(rand(from_default_1, 3)) == Array(rand(from_default_2, 3))
    end

    @testset "exponential from uniform variates" begin
        @test iszero(HeterogeneousComputing._neglog_uniform(0.0f0))
        @test iszero(HeterogeneousComputing._neglog_uniform(1.0f0))
        @test HeterogeneousComputing._neglog_uniform(0.5) ≈ log(2)
    end
end
