# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

# From the repository root, set up and run the OpenCL tests with
# julia --project=test/opencl -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
# julia --project=test/opencl test/opencl/runtests.jl
using HeterogeneousComputing, Test
@test realpath(pkgdir(HeterogeneousComputing)) == realpath(joinpath(@__DIR__, "..", ".."))

ENV["POCL_DEVICES"] = "pthread pthread"
using pocl_jll, OpenCL
using MLDataDevices: OpenCLDevice
using Adapt: adapt

@testset "OpenCL contexts" begin
    platform = only(filter(p -> occursin("Portable Computing", p.name), cl.platforms()))
    d0, d1 = cl.devices(platform)
    src_ctx = cl.Context(d0)
    contexts = (cl.Context(d0), cl.Context(d1))
    targets = map(ctx -> cl.context!(() -> DeviceUnit(OpenCLDevice()), ctx), contexts)
    cl.context!(src_ctx) do
        src_queue = cl.queue!(cl.CmdQueue())
        a = CLArray(Float32[1, 2, 3])
        src = get_compute_unit(a)
        @test src == DeviceUnit(OpenCLDevice())
        for (ctx, target) in zip(contexts, targets)
            @test target != src
            b = allocate_array(target, Float32, 3)
            @test OpenCL.context(b) == ctx
            @test get_compute_unit(b) == target
            @test get_compute_unit((a, b)) isa MixedComputeSystem
            moved = adapt(target, a)
            @test OpenCL.context(moved) == ctx
            @test adapt(target, moved) === moved
            @test adapt(CPUnit(), moved) == Float32[1, 2, 3]
            @test on_device(cl.context, target)() == ctx
            @test on_device(cl.queue, target)().context == ctx
            @test OpenCL.context(rand(rootrng(target, 42), Float32, 3)) == ctx
            f = on_device(x -> x .+ 1, target, a)
            result = f(a)
            @test OpenCL.context(result) == src_ctx
            @test Array(result) == Float32[2, 3, 4]
            @test cl.context() == src_ctx && cl.device() == d0
            @test cl.queue() == src_queue
            @test_throws ErrorException on_device(() -> error("callback exited"), target)()
            @test cl.context() == src_ctx && cl.queue() == src_queue
        end
    end
end
