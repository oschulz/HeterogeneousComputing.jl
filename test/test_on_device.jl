# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

using MLDataDevices: CPUDevice
using JLArrays: JLArray, JLBackend

@testset "test_on_device" begin
    f(x, y) = sum(x .* y)
    dummy_x, dummy_y = rand(Float32, 10), rand(Float32, 10)

    device = CPUDevice()
    @test @inferred(on_device(f, device, dummy_x, dummy_y)) isa Function
    g = on_device(f, device, dummy_x, dummy_y)

    x, y = rand(Float32, 10), rand(Float32, 10)
    @test @inferred(g(x, y)) ≈ f(x, y)
    @test_throws ArgumentError g(x)
    # Arguments on different compute units:
    @test_throws ArgumentError g(x, JLArray(y))

    h = on_device(f, CPUnit(), dummy_x, dummy_y)
    @test h(x, y) ≈ f(x, y)

    # Results are adapted back to the compute unit of the arguments:
    jl_f = on_device((x, y) -> x .* y, AbstractComputeUnit(JLBackend()), dummy_x, dummy_y)
    @test jl_f(x, y) isa Vector{Float32} && jl_f(x, y) ≈ x .* y
    # Contexts are arguments on their compute unit as well:
    ctx = GenContext{Float32}()
    jl_rand = on_device(c -> rand(c, 3), AbstractComputeUnit(JLBackend()), ctx)
    @test jl_rand(ctx) isa Vector{Float32}
    jl_ones = on_device(() -> fill_array(AbstractComputeUnit(JLBackend()), 1.0f0, 3), AbstractComputeUnit(JLBackend()))
    @test jl_ones() isa JLArray{Float32}
end
