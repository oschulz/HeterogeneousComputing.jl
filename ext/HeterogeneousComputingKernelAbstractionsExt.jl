# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingKernelAbstractionsExt

import KernelAbstractions
import KernelAbstractions.Backend as _KA_Backend

using HeterogeneousComputing
import HeterogeneousComputing: ka_backend


_KA_Backend(cunit::AbstractComputeUnit) = ka_backend(cunit)::_KA_Backend
Base.convert(::Type{_KA_Backend}, cunit::AbstractComputeUnit) = ka_backend(cunit)::_KA_Backend

ka_backend(cunit::DeviceUnit) = KernelAbstractions.get_backend(allocate_array(cunit, UInt8, 0))
ka_backend(::CPUnit) = KernelAbstractions.CPU()
KernelAbstractions.CPU(cunit::CPUnit) = ka_backend(cunit)
Base.convert(::Type{KernelAbstractions.CPU}, cunit::CPUnit) = ka_backend(cunit)

end # module HeterogeneousComputingKernelAbstractionsExt
