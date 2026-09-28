# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

using Random
using StructArrays


@testset "precision" begin
    data = gen_testdata()
    f = gen_testclosure()

    @test @inferred(get_precision(Float16(4.2))) == Float16
    @test @inferred(get_precision(Float32(4.2))) == Float32
    @test @inferred(get_precision(Float64(4.2))) == Float64
    @test @inferred(get_precision(())) == HeterogeneousComputing.NoPrecision{Tuple{}}
    @test @inferred(get_precision(42)) == HeterogeneousComputing.NoPrecision{Int}
    @test @inferred(get_precision("Hello, World!")) == HeterogeneousComputing.NoPrecision{String}
    @test @inferred(get_precision(:my_symbol)) == HeterogeneousComputing.NoPrecision{Symbol}
    @test @inferred(get_precision(Any[4.2])) == HeterogeneousComputing.NoPrecision{Any}
    @test get_precision((a = Real[1], b = rand(Float32, 2))) == Float32
    @test get_precision(Ref{Union{Nothing,Float32}}(nothing)) ==
          HeterogeneousComputing.NoPrecision{Union{Nothing,Float32}}

    # @test @inferred(get_precision(data)) == Float32
    @test get_precision(data) == Float32
    # @test @inferred(get_precision(f)) == Float32
    @test get_precision(f) == Float32
end
