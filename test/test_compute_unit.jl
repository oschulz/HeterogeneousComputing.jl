# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

using ArraysOfArrays, FillArrays, StructArrays
using Adapt
using KernelAbstractions
using MLDataDevices: CPUDevice, MetalDevice, cpu_device, get_device, default_device_rng
using JLArrays
using Random



struct _TestDeviceRNG{S} <: AbstractRNG
    state::S
end

mutable struct _TestNode
    data::Vector{Float64}
    next::_TestNode
    _TestNode(data) = new(data)
end


@testset "compute_unit" begin
    @testset "CPUnit" begin
        @test CPUnit() isa DeviceUnit{<:CPUDevice}
        @test DeviceUnit(CPUDevice()) === CPUnit()
        @test DeviceUnit(CPUDevice()) == CPUnit() && hash(DeviceUnit(CPUDevice())) == hash(CPUnit())
        @test AbstractComputeUnit(cpu_device(Float32)) === CPUnit()
        @test convert(AbstractComputeUnit, CPUDevice()) === CPUnit()
        @test get_device(CPUnit()) === CPUDevice{Nothing}()
        @test is_host_unit(CPUnit())
        @test default_device_rng(CPUnit()) === Random.default_rng()
        @test sprint(show, CPUnit()) == "CPUnit()"
        # Devices whose package isn't loaded:
        @test_throws ArgumentError DeviceUnit(MetalDevice())
        test_cunit(CPUnit())
        @test @inferred(KernelAbstractions.Backend(CPUnit())) isa KernelAbstractions.CPU
        @test @inferred(convert(KernelAbstractions.Backend, CPUnit())) isa KernelAbstractions.CPU
    end

    @testset "JLArrays" begin
        jl_unit = AbstractComputeUnit(JLBackend())
        @test !is_host_unit(jl_unit)
        @test get_compute_unit(JLArray(rand(3))) === jl_unit
        @test get_compute_unit(default_device_rng(jl_unit)) === jl_unit
        test_cunit(jl_unit)
        @test @inferred(KernelAbstractions.Backend(jl_unit)) isa JLBackend
    end

    @testset "get_compute_unit" begin
        cpu_data = StructArray(
            a = rand(Float64, 100),
            b = VectorOfSimilarVectors(rand(Float32, 10, 100)),
            c = Fill(1.5, 100)
        )

        bitstype_data = StructArray(
            a = Fill(7.2, 100),
            b = VectorOfSimilarVectors(Fill(4.2, 10, 100)),
            c = Fill(1.5, 100)
        )

        @test @inferred(get_compute_unit(cpu_data)) == CPUnit()
        @test @inferred(get_compute_unit(bitstype_data)) == ComputeUnitIndependent()
        @test get_compute_unit(Random.default_rng()) == ComputeUnitIndependent()
        @test get_compute_unit(CPUnit()) === CPUnit()

        # StructArrays can't adapt nested array columns, use flat columns:
        flat_data = StructArray(a = rand(Float64, 10), b = rand(Float32, 10))
        jl_data = adapt(AbstractComputeUnit(JLBackend()), flat_data)
        @test get_compute_unit(jl_data) === AbstractComputeUnit(JLBackend())
        @test get_compute_unit((flat_data, jl_data)) === MixedComputeSystem()
        @test get_compute_unit(x -> jl_data.a .* x) === AbstractComputeUnit(JLBackend())

        # Nested array wrappers, inferred before their components (with element
        # types not used elsewhere, so no inference results are cached):
        nested = VectorOfSimilarVectors(reshape(view(reshape(view(zeros(Int8, 16), :), 4, 4), :, 1:2), 2, 4))
        @test @inferred(get_compute_unit(nested)) === CPUnit()
        @test @inferred(get_compute_unit(StructArray(a = nested, b = view(rand(UInt8, 8), 1:4)))) === CPUnit()

        A = reshape(view(zeros(4), :), 2, 2)
        @test @inferred(get_compute_unit(A)) === CPUnit()
        @test @inferred(get_compute_unit(VectorOfSimilarVectors(A))) === CPUnit()
        @test @inferred(get_compute_unit((a = A, b = (Fill(1.0, 3), view(rand(4), 2:3))))) === CPUnit()

        # Types and RNGs:
        @test @inferred(get_compute_unit(Float64)) === ComputeUnitIndependent()
        @test @inferred(get_compute_unit(Xoshiro(42))) === ComputeUnitIndependent()
        @test get_compute_unit(MersenneTwister(42)) === CPUnit()
        jl_rng = _TestDeviceRNG(JLArray(rand(UInt64, 4)))
        @test get_compute_unit(jl_rng) === AbstractComputeUnit(JLBackend())
        @test get_compute_unit((jl_rng, rand(3))) === MixedComputeSystem()

        # Compute units and contexts:
        @test @inferred(get_compute_unit((x = rand(Float32, 3), ctx = GenContext()))) === CPUnit()
        jl_ctx = GenContext(AbstractComputeUnit(JLBackend()))
        @test get_compute_unit((x = 4.2, ctx = jl_ctx)) === AbstractComputeUnit(JLBackend())
        @test get_compute_unit(Dict(:a => 1)) === CPUnit()
    end

    @testset "get_compute_unit with reference loops" begin
        # Self-referencing closure, captured via a Core.Box:
        local f
        x = JLArray(rand(3))
        f = n -> n > 0 ? f(n - 1) : sum(x)
        @test get_compute_unit(f) === AbstractComputeUnit(JLBackend())

        a = _TestNode(rand(2))
        @test get_compute_unit(a) === CPUnit()  # undefined field
        b = _TestNode(rand(2))
        a.next = b
        b.next = a
        @test get_compute_unit(a) === CPUnit()
        @test get_compute_unit([a]) === CPUnit()
    end

    @testset "merge_compute_units" begin
        jl_unit = AbstractComputeUnit(JLBackend())
        @test merge_compute_units() === ComputeUnitIndependent()
        @test merge_compute_units(CPUnit(), CPUnit()) === CPUnit()
        @test merge_compute_units(CPUnit(), ComputeUnitIndependent()) === CPUnit()
        @test merge_compute_units(ComputeUnitIndependent(), jl_unit, jl_unit) === jl_unit
        @test merge_compute_units(CPUnit(), jl_unit) === MixedComputeSystem()
    end
end
