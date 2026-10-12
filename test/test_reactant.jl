# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

# Reactant isn't a static test dependency (it only supports 64-bit Linux and
# macOS), runtests.jl adds it on the fly where supported. The backend
# defaults to the CPU, set the environment variable
# `HETEROGENEOUSCOMPUTING_REACTANT_BACKEND` (e.g. to "gpu") to change it.

using HeterogeneousComputing
using Test

# Two CPU devices for multi-device tests (only effective before Reactant
# initializes its CPU client):
ENV["XLA_FLAGS"] = get(ENV, "XLA_FLAGS", "") * " --xla_force_host_platform_device_count=2"

using Reactant
using MLDataDevices: ReactantDevice, reactant_device, get_device
using Adapt: adapt
using Random
using KernelAbstractions

Reactant.set_default_backend(get(ENV, "HETEROGENEOUSCOMPUTING_REACTANT_BACKEND", "cpu"))

@testset "Reactant" begin
    x = Reactant.to_rarray(rand(Float32, 5))
    cunit = get_compute_unit(x)
    @test cunit isa DeviceUnit{<:ReactantDevice}
    @test DeviceUnit(ReactantDevice()) == cunit
    @test get_compute_unit(Reactant.to_rarray(2.0f0; track_numbers = Number)) == cunit
    @test get_precision(Reactant.to_rarray(2.0f0; track_numbers = Number)) === Float32

    A = allocate_array(cunit, Float32, 2, 3)
    @test A isa Reactant.ConcreteRArray{Float32,2} && size(A) == (2, 3)
    @test Array(fill_array(cunit, 1.5, 2)) == [1.5, 1.5]

    @test KernelAbstractions.Backend(cunit) isa KernelAbstractions.Backend

    ctx = GenContext{Float32}(cunit)
    @test get_compute_unit(get_rng(ctx)) == cunit

    # Units derived within compiled code don't specify the device, but merge
    # with units that do:
    function _partial_unit_merges(partial, concrete)
        return partial != concrete && merge_compute_units(partial, concrete) === concrete &&
               merge_compute_units(concrete, partial) === concrete
    end

    function f(ctx, x)
        return (
            randn(ctx, 3), randexp!(ctx, similar(x)), fill_array(ctx, sum(x) * 2.0, 2), rand(ctx),
            _partial_unit_merges(get_compute_unit(x), get_compute_unit(ctx)),
            get_precision(sum(x))
        )
    end
    f_compiled = @compile f(ctx, x)
    r1 = f_compiled(ctx, x)
    r2 = f_compiled(ctx, x)
    @test r1[1] isa Reactant.ConcreteRArray{Float32,1} && size(r1[1]) == (3,)
    @test all(>=(0), Array(r1[2]))
    # Floating point fills use the context precision:
    @test r1[3] isa Reactant.ConcreteRArray{Float32,1} && Array(r1[3]) ≈ fill(2 * sum(Array(x)), 2)
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

    devs = Reactant.devices()
    if length(devs) >= 2
        @testset "multiple devices" begin
            u0 = DeviceUnit(reactant_device(; device = devs[1]))
            u1 = DeviceUnit(reactant_device(; device = devs[2]))
            @test u0 != u1
            @test u0 == get_compute_unit(Reactant.to_rarray(rand(2); device = devs[1]))
            # Unspecified clients and devices are resolved like Reactant does:
            @test DeviceUnit(ReactantDevice()) == u0
            @test DeviceUnit(reactant_device(; client = Reactant.XLA.client(devs[2]))) == u0

            # Sharded arrays aren't located on any specific device:
            mesh = Reactant.Sharding.Mesh(reshape(devs[1:2], 2), (:x,))
            s = Reactant.to_rarray(rand(4); sharding = Reactant.Sharding.NamedSharding(mesh, (:x,)))
            a0 = Reactant.to_rarray(rand(2); device = devs[1])
            a1 = Reactant.to_rarray(rand(2); device = devs[2])
            @test get_compute_unit(s) != u0 && get_compute_unit(s) != u1
            @test get_compute_unit((a0, s, a1)) === MixedComputeSystem()
            # Units of sharded arrays keep the device mesh, new data on them
            # is replicated over it:
            us = get_compute_unit(s)
            @test us ==
                  get_compute_unit(Reactant.to_rarray(rand(6); sharding = Reactant.Sharding.NamedSharding(mesh, (:x,))))
            @test get_compute_unit(allocate_array(us, Float32, 4)) == us
            @test get_compute_unit(adapt(us, rand(4))) == us
            ctx_s = GenContext{Float32}(us)
            @test get_compute_unit(get_rng(ctx_s)) == us
            sy = Reactant.to_rarray(rand(Float32, 4); sharding = Reactant.Sharding.NamedSharding(mesh, (:x,)))
            @test @jit(((ctx, y) -> y .+ randn(ctx, size(y)))(ctx_s, sy)) isa Reactant.ConcreteRArray{Float32,1}


            ctx0 = GenContext{Float32}(u0)
            @test get_compute_unit(get_rng(GenContext{Float32}(u1))) == u1
            @test adapt(u0, ctx0).rng === get_rng(ctx0)
            ctx1 = adapt(u1, ctx0)
            @test get_compute_unit(ctx1) == u1 && get_compute_unit(get_rng(ctx1)) == u1
            y1 = Reactant.to_rarray(rand(Float32, 3); device = devs[2])
            @test @jit(((ctx, y) -> y .+ randn(ctx, size(y)))(ctx1, y1)) isa Reactant.ConcreteRArray{Float32,1}
        end
    end

    g = on_device((x, y) -> sum(x .* y), cunit, rand(Float32, 3), rand(Float32, 3))
    @test g(ones(Float32, 3), ones(Float32, 3)) ≈ 3
end
