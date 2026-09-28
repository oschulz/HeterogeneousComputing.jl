# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

"""
    HeterogeneousComputing

Tools for heterogeneous computing in Julia.
"""
module HeterogeneousComputing

using Random
using Base: AbstractLock

using MLDataDevices: MLDataDevices, AbstractDevice, CPUDevice, UnknownDevice
using MLDataDevices: CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice, OpenCLDevice, ReactantDevice
using MLDataDevices: get_device, default_device_rng
using Adapt: Adapt, adapt
import GPUArraysCore

include("precision.jl")
include("rng.jl")
include("compute_unit.jl")
include("gen_context.jl")
include("numtype.jl")
include("on_device.jl")

end # module
