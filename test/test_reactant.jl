# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

# Reactant isn't a static test dependency (it only supports 64-bit Linux and
# macOS), runtests.jl adds it on the fly where supported. The backend
# defaults to the CPU, set the environment variable
# `HETEROGENEOUSCOMPUTING_REACTANT_BACKEND` (e.g. to "gpu") to change it.

using HeterogeneousComputing
using Test

using Reactant
using MLDataDevices: ReactantDevice, get_device
using Random
using KernelAbstractions

Reactant.set_default_backend(get(ENV, "HETEROGENEOUSCOMPUTING_REACTANT_BACKEND", "cpu"))

@testset "Reactant" begin
    x = Reactant.to_rarray(rand(Float32, 5))
    cunit = get_compute_unit(x)
    @test cunit isa DeviceUnit{<:ReactantDevice}
    # Units derived within compiled code don't specify the device, but merge
    # with units that do:
    partial = DeviceUnit(ReactantDevice())
    @test partial != cunit
    @test merge_compute_units(cunit, partial) === cunit
    @test merge_compute_units(partial, cunit) === cunit
    @test get_compute_unit(Reactant.to_rarray(2.0f0; track_numbers = Number)) == cunit
    @test get_precision(Reactant.to_rarray(2.0f0; track_numbers = Number)) === Float32

    A = allocate_array(cunit, Float32, 2, 3)
    @test A isa Reactant.ConcreteRArray{Float32,2} && size(A) == (2, 3)
    @test Array(fill_array(cunit, 1.5, 2)) == [1.5, 1.5]

    @test KernelAbstractions.Backend(cunit) isa KernelAbstractions.Backend

    ctx = GenContext{Float32}(cunit)
    @test get_compute_unit(get_rng(ctx)) == cunit

    function f(ctx, x)
        return (
            randn(ctx, 3), randexp!(ctx, similar(x)), fill_array(ctx, sum(x), 2), rand(ctx),
            merge_compute_units(get_compute_unit(x), get_compute_unit(ctx)) == get_compute_unit(ctx),
            get_precision(sum(x))
        )
    end
    f_compiled = @compile f(ctx, x)
    r1 = f_compiled(ctx, x)
    r2 = f_compiled(ctx, x)
    @test r1[1] isa Reactant.ConcreteRArray{Float32,1} && size(r1[1]) == (3,)
    @test all(>=(0), Array(r1[2]))
    @test Array(r1[3]) ≈ fill(sum(Array(x)), 2)
    @test r1[5] == true
    @test 0 <= Float32(r1[4]) <= 1
    @test Float32(r1[4]) != Float32(r2[4])
    @test r1[6] === Float32
    # The RNG state advances across calls of the compiled function:
    @test Array(r1[1]) != Array(r2[1])
    @test Array(r1[2]) != Array(r2[2])

    # RNGs must be passed into compiled code, draws only happen there:
    @test_throws ArgumentError @jit((x -> randn(GenContext{Float32}(get_compute_unit(x)), 3))(x))
    @test_throws ArgumentError rand(ctx, 3)
    @test_throws ArgumentError rand(ctx)

    g = on_device((x, y) -> sum(x .* y), cunit, rand(Float32, 3), rand(Float32, 3))
    @test g(ones(Float32, 3), ones(Float32, 3)) ≈ 3
end
