# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

using Random

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

    _check_array(A, ::Type{T}, sz::Dims{N}) where {T,N} = @test A isa AbstractArray{T,N} && size(A) == sz

    _check_array(@inferred(allocate_array(ctx, (4, 5))), Float32, (4, 5))
    _check_array(@inferred(allocate_array(ctx, 4, 5)), Float32, (4, 5))
    _check_array(@inferred(allocate_array(ctx, Float16, (4, 5))), Float16, (4, 5))
    _check_array(@inferred(allocate_array(ctx, Float16, 4, 5)), Float16,(4, 5))

    for randfun in (rand, randn, randexp)
        @testset "$randfun" begin
            @inferred(randfun(ctx)) isa Float32
            _check_array(@inferred(randfun(ctx, (4, 5))), Float32, (4, 5))
            _check_array(@inferred(randfun(ctx, 4, 5)), Float32, (4, 5))
        end
    end
end
