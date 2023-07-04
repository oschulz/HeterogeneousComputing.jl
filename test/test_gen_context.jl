# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

using Random

@testset "gen_context" begin
    RNG = typeof(Random.default_rng())

    @test @inferred(GenContext()) isa GenContext{Float64,RNG,CPUnit}
    @test @inferred(GenContext(Random.default_rng())) isa GenContext{Float64,RNG,CPUnit}
    @test @inferred(GenContext{Float32}()) isa GenContext{Float32,RNG,CPUnit}

    ctx = GenContext{Float32}()

    @test typeof(@inferred(GenContext(ctx))) == typeof(ctx)
    @test typeof(@inferred(GenContext{Float16}(ctx))) == typeof(GenContext{Float16}())
    @test typeof(@inferred(convert(GenContext{Float16}, ctx))) == typeof(GenContext{Float16}())
    @test typeof(@inferred((typeof(ctx))(ctx))) == typeof(ctx)

    _check_array(A, ::Type{T}, sz::Dims{N}) where {T,N} = @test A isa AbstractArray{T,N} && size(A) == sz

    _check_array(@inferred(allocate_array(ctx, (4, 5))), Float32, (4, 5))
    _check_array(@inferred(allocate_array(ctx, 4, 5)), Float32, (4, 5))
    _check_array(@inferred(allocate_array(ctx, Float16, (4, 5))), Float16, (4, 5))
    _check_array(@inferred(allocate_array(ctx, Float16, 4, 5)), Float16,(4, 5))

    for randfun in (rand, randn, randexp)
        @inferred(rand(ctx)) isa Float32
        @inferred(rand(ctx, Float16)) isa Float16
        _check_array(@inferred(rand(ctx, (4, 5))), Float32, (4, 5))
        _check_array(@inferred(rand(ctx, 4, 5)), Float32, (4, 5))
        _check_array(@inferred(rand(ctx, Float16, (4, 5))), Float16, (4, 5))
        _check_array(@inferred(rand(ctx, Float16, 4, 5)), Float16, (4, 5))
    end

    for randfun! in (rand!, randn, randexp!)
        A = Array{Float32}(undef, 4, 5)
        @test @inferred(rand!(ctx, A)) === A
    end
end
