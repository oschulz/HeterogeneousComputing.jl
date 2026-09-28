# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

using Random
using MLDataDevices: CPUDevice, cpu_device
using JLArrays



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

    _check_array(A, ::Type{T}, sz::Dims{N}) where {T,N} = @test A isa AbstractArray{T,N} && size(A) == sz

    _check_array(@inferred(allocate_array(ctx, (4, 5))), Float32, (4, 5))
    _check_array(@inferred(allocate_array(ctx, 4, 5)), Float32, (4, 5))
    _check_array(@inferred(allocate_array(ctx, Float16, (4, 5))), Float16, (4, 5))
    _check_array(@inferred(allocate_array(ctx, Float16, 4, 5)), Float16, (4, 5))
    @test @inferred(fill_array(ctx, 1.5f0, 2, 3)) == fill(1.5f0, 2, 3)

    test_gencontext_draws(ctx)
    test_gencontext_draws(GenContext{Float64}(Xoshiro(123)))

    @testset "JLArrays" begin
        jl_unit = AbstractComputeUnit(JLBackend())
        jl_ctx = GenContext{Float32}(jl_unit)
        @test jl_ctx isa GenContext{Float32,typeof(jl_unit)}
        @test GenContext{Float32}(get_rng(jl_ctx)) === jl_ctx
        @test get_gencontext(allocate_array(jl_ctx, 3)) isa GenContext{Float32,typeof(jl_unit)}
        test_gencontext_draws(jl_ctx)
    end

    @testset "exponential from uniform variates" begin
        @test iszero(HeterogeneousComputing._neglog_uniform(0.0f0))
        @test iszero(HeterogeneousComputing._neglog_uniform(1.0f0))
        @test HeterogeneousComputing._neglog_uniform(0.5) ≈ log(2)
    end
end
