# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

using MLDataDevices: CPUDevice

@testset "test_on_device" begin
    f(x, y) = sum(x .* y)
    dummy_x, dummy_y = rand(Float32, 10), rand(Float32, 10)

    device = CPUDevice()
    @test @inferred(on_device(f, device, dummy_x, dummy_y) isa Function)
    g = on_device(f, device, dummy_x, dummy_y)

    x, y = rand(Float32, 10), rand(Float32, 10)
    @test @inferred(g(x, y)) ≈ f(x, y)
    @test_throws ArgumentError g(x)

    h = on_device(f, CPUnit(), dummy_x, dummy_y)
    @test h(x, y) ≈ f(x, y)
end
