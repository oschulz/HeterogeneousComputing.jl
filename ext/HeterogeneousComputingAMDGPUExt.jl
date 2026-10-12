# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingAMDGPUExt

import AMDGPU

using HeterogeneousComputing: DeviceUnit
import HeterogeneousComputing: _canonical_device, _within_unit

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

end # module HeterogeneousComputingAMDGPUExt
