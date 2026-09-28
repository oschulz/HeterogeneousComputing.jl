# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

using ArraysOfArrays, FillArrays, StructArrays
using Adapt
using KernelAbstractions
using MLDataDevices: CPUDevice, cpu_device, get_device, default_device_rng
using JLArrays
using Random



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

        @test (get_compute_unit(cpu_data)) == CPUnit() # @inferred
        @test @inferred(get_compute_unit(bitstype_data)) == ComputeUnitIndependent()
        @test get_compute_unit(Random.default_rng()) == ComputeUnitIndependent()
        @test get_compute_unit(CPUnit()) === CPUnit()

        # StructArrays can't adapt nested array columns, use flat columns:
        flat_data = StructArray(a = rand(Float64, 10), b = rand(Float32, 10))
        jl_data = adapt(AbstractComputeUnit(JLBackend()), flat_data)
        @test get_compute_unit(jl_data) === AbstractComputeUnit(JLBackend())
        @test get_compute_unit((flat_data, jl_data)) === MixedComputeSystem()
        @test get_compute_unit(x -> jl_data.a .* x) === AbstractComputeUnit(JLBackend())
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
