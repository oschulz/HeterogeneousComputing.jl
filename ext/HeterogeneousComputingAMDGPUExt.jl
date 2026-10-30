# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingAMDGPUExt

import AMDGPU
import GPUArrays

using HeterogeneousComputing: DeviceUnit
import HeterogeneousComputing: _canonical_device, _within_unit

import MLDataDevices
using MLDataDevices: AMDGPUDevice


const AMDGPUUnit = DeviceUnit{<:AMDGPUDevice}

_canonical_device(::AMDGPUDevice{Nothing}) = _canonical_device(AMDGPUDevice(AMDGPU.device()))

function _within_unit(f, cunit::AMDGPUUnit)
    old_dev = AMDGPU.device()
    AMDGPU.device!(cunit.device.device)
    try
        return f()
    finally
        AMDGPU.device!(old_dev)
    end
end

# rocRAND generators are bound to the device they were created on without
# recording it, so contexts use the GPUArrays RNG, which generates on the
# device of the array it fills (like CUDA v6 does by default):
MLDataDevices.default_device_rng(::AMDGPUUnit) = GPUArrays.RNG{AMDGPU.ROCArray}()

end # module HeterogeneousComputingAMDGPUExt
