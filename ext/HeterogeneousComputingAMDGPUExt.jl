# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingAMDGPUExt

import AMDGPU
import Random

using HeterogeneousComputing: AbstractComputeUnit, DeviceUnit
import HeterogeneousComputing: _canonical_device, _within_unit, _seeded_rng_like

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

function _seeded_rng_like(::AMDGPU.rocRAND.RNG, ::AbstractComputeUnit, seeds::NTuple{4,UInt64})
    rng = AMDGPU.rocRAND.RNG()
    Random.seed!(rng, seeds[1])
    return rng
end

end # module HeterogeneousComputingAMDGPUExt
