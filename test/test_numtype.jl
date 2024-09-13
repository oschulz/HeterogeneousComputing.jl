# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

import Dates

@testset "numtype" begin
    @testset "real_numtype" begin
        a = 4.2f0
        b = Complex(a, a)
        A = fill(fill(rand(Float32, 5), 10), 5)
        B = fill(rand(Float32, 5), 10)
        nt = (a = 4.2f0, b = 42)
        tpl = (4.2f0, Float16(2.3))
        deepnt = (a = a, b = b, A = A, B = B, nt = nt, tpl = tpl)
        for x in [a, A, B, nt, tpl, deepnt]
            @test @inferred (real_numtype(typeof(x))) == Float32
        end

        for x in [nothing, missing, (), :foo, "foo", Dates.TWENTYFOURHOUR]
            @test @inferred(real_numtype(typeof(x))) == Bool
        end
    end
end
