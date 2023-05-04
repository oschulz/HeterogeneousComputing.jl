# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

using ArraysOfArrays, FillArrays, StructArrays
using Adapt
using KernelAbstractions


@static if isdefined(KernelAbstractions, :Backend)
    import KernelAbstractions.Backend as _KA_Backend
else
    # KernelAbstractions < v0.9:
    import KernelAbstractions.Device as _KA_Backend
end


@testset "compute_unit" begin
    cpu_data = StructArray(
        a = rand(Float64, 100),
        b = nestedview(rand(Float32, 10, 100)),
        c = Fill(1.5, 100),
    )

    bitstype_data = StructArray(
        a = Fill(7.2, 100),
        b = nestedview(Fill(4.2, 10, 100),),
        c = Fill(1.5, 100),
    )

    @test #=@inferred=#(get_compute_unit(cpu_data)) == CPUnit()
    @test @inferred(get_compute_unit(bitstype_data)) == ComputeUnitIndependent()

    _test_cunit(cunit) = @testset "$(nameof(typeof(cunit)))" begin
        @test @inferred(get_total_memory(cunit)) isa Integer
        @test 0 < get_total_memory(cunit)
        @test @inferred(get_free_memory(cunit)) isa Integer
        @test 0 <= get_free_memory(cunit) <= get_total_memory(cunit)

        @test @inferred(_KA_Backend(cunit)) isa _KA_Backend
        @test @inferred(convert(_KA_Backend, cunit)) isa _KA_Backend
    end

    _test_cunit(CPUnit())

    if isdefined(Main, :CUDA)
        @testset "CUDA" begin
            using CUDA
            cuda_unit = AbstractComputeUnit(CuDevice(0))
            cuda_data = adapt(cuda_unit, cpu_data)
            @test #=@inferred=#(get_compute_unit(cuda_data)) == cuda_unit
            _test_cunit(cuda_unit)
        end
    end
end
