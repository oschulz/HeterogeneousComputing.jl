# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

import ArraysOfArrays, FillArrays, StructArrays
using Test
using HeterogeneousComputing
using Random: rand!, randn!, randexp!
using Adapt: adapt


function gen_testdata()
    return (
        x = rand(Float32),
        A = rand(Float16, 3, 4, 5),
        str = "Hello, World!",
        sym = :SomeSymbol,
        sa = StructArrays.StructArray((
            a = rand(Float16, 100),
            b = ArraysOfArrays.VectorOfSimilarVectors(rand(Float32, 10, 100)),
            c = FillArrays.Fill(Float16(1.5), 100),
            d = rand(-7:15, 100)
        ))
    )
end


function gen_testclosure()
    data = gen_testdata()
    return function _testclosure(args...)
        return merge(data, (args = args,))
    end
end


_on_host(A) = Array(A)
_on_host(x::Number) = x

# Tests the compute unit contract of `cunit`:
function test_cunit(cunit::AbstractComputeUnit)
    @testset "$cunit" begin
        @test @inferred(get_total_memory(cunit)) isa Integer
        @test 0 < get_total_memory(cunit)
        @test @inferred(get_free_memory(cunit)) isa Integer
        @test 0 <= get_free_memory(cunit) <= get_total_memory(cunit)

        A = @inferred(allocate_array(cunit, Float32, (4, 5)))
        @test eltype(A) == Float32 && size(A) == (4, 5)
        @test get_compute_unit(A) == cunit
        @test typeof(@inferred(allocate_array(cunit, Float32, 4, 5))) == typeof(A)

        B = @inferred(fill_array(cunit, 4.2, 2, 3))
        @test get_compute_unit(B) == cunit
        @test _on_host(B) == fill(4.2, 2, 3)

        # adapt preserves numerical precision:
        x = rand(Float64, 7)
        x_adapted = adapt(cunit, x)
        @test eltype(x_adapted) == Float64 && get_compute_unit(x_adapted) == cunit
        @test _on_host(x_adapted) == x
        @test adapt(CPUnit(), x_adapted) == x
    end
end

# Tests random number generation with `ctx`:
function test_gencontext_draws(ctx::GenContext{T}) where T
    cunit = get_compute_unit(ctx)
    @testset "draws with $(typeof(ctx))" begin
        for (randfun, randfun!) in ((rand, rand!), (randn, randn!), (randexp, randexp!))
            @test @inferred(randfun(ctx)) isa T
            A = @inferred(randfun(ctx, (4, 5)))
            @test A isa AbstractArray{T,2} && size(A) == (4, 5)
            @test get_compute_unit(A) == cunit
            @test typeof(@inferred(randfun(ctx, 4, 5))) == typeof(A)
            @test @inferred(randfun!(ctx, A)) === A
        end

        n = 10^5
        U = _on_host(rand(ctx, n))
        @test all(x -> 0 <= x <= 1, U)
        @test sum(U) / n ≈ 0.5 atol = 0.01
        N = _on_host(randn(ctx, n))
        @test sum(N) / n ≈ 0 atol = 0.02
        E = _on_host(randexp(ctx, n))
        @test all(x -> 0 <= x < Inf, E)
        @test sum(E) / n ≈ 1 atol = 0.02
    end
end
