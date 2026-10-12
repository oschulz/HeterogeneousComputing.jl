# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

# CUDA tests, not part of the default test suite (they need a CUDA GPU).
# Run with `julia --project=test/cuda test/cuda/runtests.jl` after
# instantiating that project.

using HeterogeneousComputing
using Test

using CUDA
using KernelAbstractions
using MLDataDevices: CUDADevice, get_device, default_device_rng
using StructArrays
using Random
using Adapt: adapt

include(joinpath(@__DIR__, "..", "testutils.jl"))

CUDA.allowscalar(false)

@testset "CUDA" begin
    cunit = AbstractComputeUnit(CUDA.device())
    @test cunit isa DeviceUnit{<:CUDADevice}
    @test isbits(cunit)
    @test DeviceUnit(CUDADevice()) === cunit
    @test !is_host_unit(cunit)
    test_cunit(cunit)
    @test @inferred(KernelAbstractions.Backend(cunit)) isa CUDA.CUDABackend

    x = CUDA.rand(Float32, 10)
    @test get_compute_unit(x) === cunit
    @test get_compute_unit(view(x, 2:5)) === cunit
    @test get_compute_unit(StructArray(a = x, b = CUDA.rand(Float32, 10))) === cunit
    @test get_compute_unit((x, rand(10))) === MixedComputeSystem()
    @test get_compute_unit(default_device_rng(cunit)) === cunit

    ctx = GenContext{Float32}(cunit)
    @test get_rng(ctx) isa typeof(CUDA.default_rng())
    @test GenContext{Float32}(get_rng(ctx)) === ctx
    @test GenContext{Float32}(CUDADevice()) isa typeof(ctx)
    # cuRAND library RNGs are bound to a CUDA context:
    library_rng() = isdefined(CUDA, :cuRAND) ? CUDA.cuRAND.LibraryRNG() : CUDA.CURAND.RNG()
    @test GenContext{Float32}(library_rng()).cunit === cunit
    test_gencontext_draws(GenContext{Float32}(library_rng()))
    # Contexts moved to other units get RNGs derived from theirs:
    @test adapt(CPUnit(), GenContext{Float32}(library_rng())) isa GenContext{Float32,CPUnit,Xoshiro}
    @test get_compute_unit(adapt(CPUnit(), GenContext{Float32}(cunit))) === CPUnit()
    derived_cuda() = adapt(cunit, GenContext{Float32}(Xoshiro(1)))
    # Derived RNGs use more than 32 bits of their seeds:
    seeded(seed) = GenContext{Float32}(
        cunit,
        HeterogeneousComputing._seeded_rng(cunit, (seed, zero(seed), zero(seed), zero(seed)))
    )
    @test Array(rand(seeded(UInt64(1)), 4)) != Array(rand(seeded(UInt64(1) << 32), 4))
    @test get_compute_unit(get_rng(derived_cuda())) === cunit
    @test Array(rand(derived_cuda(), 3)) == Array(rand(derived_cuda(), 3))

    test_gencontext_draws(ctx)
    test_gencontext_draws(GenContext{Float64}(cunit))

    # GPU uniform variates can be exactly one, exponential variates derived
    # from them must still be finite:
    @test all(isfinite, randexp(ctx, 10^8))
end
