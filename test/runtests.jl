# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package HeterogeneousComputing" begin
    # Reactant only supports 64-bit Linux and macOS (some of its
    # dependencies break already during precompilation elsewhere), so it
    # can't be a static test dependency. Tracing behavior is
    # platform-independent, so testing on Linux with a current Julia
    # release suffices; Julia 1.10's parallel precompilation is prone to
    # deadlock on Reactant's large extension set. Reactant is added before
    # any other test dependency is loaded, since adding it changes the
    # versions of their dependencies (LLVM via GPUArrays). The environment
    # is precompiled as a whole afterwards, since the automatic
    # precompilation of the changed packages fails for indirect
    # dependencies on Julia v1.13.2:
    test_reactant =
        Sys.islinux() && Sys.ARCH in (:x86_64, :aarch64) && isempty(VERSION.prerelease) &&
        VERSION >= v"1.11"
    if test_reactant
        import Pkg
        if Base.identify_package("Reactant") === nothing
            Pkg.add("Reactant"; allow_autoprecomp = false)
            Pkg.precompile()
        end
    end

    include("testutils.jl")
    include("test_precision.jl")
    include("test_rng.jl")
    include("test_compute_unit.jl")
    include("test_gen_context.jl")
    include("test_numtype.jl")
    include("test_on_device.jl")
    test_reactant && include("test_reactant.jl")
    include("test_aqua.jl")
    include("test_docs.jl")
end # testset
