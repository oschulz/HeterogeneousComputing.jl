# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

using Random
using MLDataDevices: CPUDevice, cpu_device
using JLArrays
import GPUArrays
using Adapt: adapt



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

    # Draws use the RNG of the context:
    @test rand(GenContext(Xoshiro(1)), 3) == rand(Xoshiro(1), 3)
    @test randn(GenContext{Float32}(Xoshiro(1))) == randn(Xoshiro(1), Float32)

    _check_array(A, ::Type{T}, sz::Dims{N}) where {T,N} = @test A isa AbstractArray{T,N} && size(A) == sz

    _check_array(@inferred(allocate_array(ctx, (4, 5))), Float32, (4, 5))
    _check_array(@inferred(allocate_array(ctx, 4, 5)), Float32, (4, 5))
    _check_array(@inferred(allocate_array(ctx, Float16, (4, 5))), Float16, (4, 5))
    _check_array(@inferred(allocate_array(ctx, Float16, 4, 5)), Float16, (4, 5))
    @test @inferred(fill_array(ctx, 1.5, 2, 3)) == fill(1.5f0, 2, 3)
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

        @test adapt(jl_unit, GenContext{Float32}()) isa GenContext{Float32,typeof(jl_unit),<:GPUArrays.RNG}
        @test get_compute_unit(adapt(CPUnit(), jl_ctx)) === CPUnit()
    end

    @testset "exponential from uniform variates" begin
        @test iszero(HeterogeneousComputing._neglog_uniform(0.0f0))
        @test iszero(HeterogeneousComputing._neglog_uniform(1.0f0))
        @test HeterogeneousComputing._neglog_uniform(0.5) ≈ log(2)
    end
end
