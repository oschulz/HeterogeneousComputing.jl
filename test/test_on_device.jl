# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using HeterogeneousComputing
using Test

import Adapt
using MLDataDevices: AbstractDevice, CPUDevice, get_device
using JLArrays: JLArray, JLBackend
using RecursiveArrayTools: DiffEqArray

struct _DeviceBox{A}
    data::A
end
Adapt.adapt_structure(dev::AbstractDevice, x::_DeviceBox) = _DeviceBox(Adapt.adapt(dev, x.data))

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

    calls = Ref(0)
    unknown, other = _UnknownUnitRNG(), JLArray(y)
    count_call(args...) = (calls[] += 1)
    counted = on_device(count_call, CPUnit(), x, other, unknown)
    for args in ((x, other, unknown), (unknown, x, other), (x, unknown, other))
        @test_throws ArgumentError counted(args...)
    end
    @test iszero(calls[])
    @test counted(unknown, x, x) == 1

    for args in (((host = x, unknown = unknown, other = other),), ((x, unknown), other), (other, Some((unknown, x))))
        nested = on_device(count_call, CPUnit(), args...)
        @test_throws ArgumentError nested(args...)
    end
    @test calls[] == 1
    compatible = (host = x, unknown = unknown)
    @test on_device(count_call, CPUnit(), compatible)(compatible) == 2

    h = on_device(f, CPUnit(), dummy_x, dummy_y)
    @test h(x, y) ≈ f(x, y)

    # Results are adapted back to the compute unit of the arguments:
    jl_f = on_device((x, y) -> x .* y, AbstractComputeUnit(JLBackend()), dummy_x, dummy_y)
    @test jl_f(x, y) isa Vector{Float32} && jl_f(x, y) ≈ x .* y
    box = _DeviceBox(Float64[1, 2])
    unit = AbstractComputeUnit(JLBackend())
    for target in (unit, get_device(unit))
        @test on_device(b -> b.data isa JLArray{Float64}, target, box)(box)
    end
    trajectory = DiffEqArray([Float16[3, 4], Float16[5, 6]], Float16[1, 2])
    @test Adapt.adapt(unit, trajectory).t === trajectory.t
    placement(v) = (first(v.u) isa JLArray{Float16}, v.t isa Vector{Float16})
    for target in (unit, get_device(unit))
        @test on_device(placement, target, trajectory)(trajectory) == (true, true)
    end
    # Contexts are arguments on their compute unit as well:
    ctx = GenContext{Float32}()
    jl_rand = on_device(c -> rand(c, 3), AbstractComputeUnit(JLBackend()), ctx)
    @test jl_rand(ctx) isa Vector{Float32}
    jl_ones = on_device(() -> fill_array(AbstractComputeUnit(JLBackend()), 1.0f0, 3), AbstractComputeUnit(JLBackend()))
    @test jl_ones() isa JLArray{Float32}
end
