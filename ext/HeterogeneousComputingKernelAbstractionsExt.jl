# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingKernelAbstractionsExt

isdefined(Base, :get_extension) ? (import KernelAbstractions) : (import ..KernelAbstractions)

using HeterogeneousComputing
import HeterogeneousComputing: ka_backend


@static if isdefined(KernelAbstractions, :Backend)
    @static if isdefined(Base, :get_extension)
        import KernelAbstractions.Backend as _KA_Backend
    else
        import ..KernelAbstractions.Backend as _KA_Backend
    end
else
    # KernelAbstractions < v0.9:
    @static if isdefined(Base, :get_extension)
        import KernelAbstractions.Device as _KA_Backend
    else
        import ..KernelAbstractions.Device as _KA_Backend
    end
end

_KA_Backend(cunit::AbstractComputeUnit) = ka_backend(cunit)::_KA_Backend
Base.convert(::Type{_KA_Backend}, cunit::AbstractComputeUnit) = ka_backend(cunit)::_KA_Backend

KernelAbstractions.GPU(cunit::AbstractGPUnit) = ka_backend(cunit)::KernelAbstractions.GPU
Base.convert(::Type{KernelAbstractions.GPU}, cunit::AbstractGPUnit) = ka_backend(cunit)::KernelAbstractions.GPU

ka_backend(::CPUnit) = KernelAbstractions.CPU()
KernelAbstractions.CPU(cunit::CPUnit) = ka_backend(cunit)
Base.convert(::Type{KernelAbstractions.CPU}, cunit::CPUnit) = ka_backend(cunit)

end # module HeterogeneousComputingKernelAbstractionsExt
