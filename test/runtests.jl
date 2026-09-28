# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package HeterogeneousComputing" begin
    include("testutils.jl")
    include("test_aqua.jl")
    include("test_precision.jl")
    include("test_rng.jl")
    include("test_compute_unit.jl")
    include("test_gen_context.jl")
    include("test_numtype.jl")
    include("test_on_device.jl")

    # Reactant only supports 64-bit Linux and macOS, so it can't be a static
    # test dependency:
    if Sys.WORD_SIZE == 64 && (Sys.islinux() || Sys.isapple()) && isempty(VERSION.prerelease)
        import Pkg
        Base.identify_package("Reactant") === nothing && Pkg.add("Reactant")
        include("test_reactant.jl")
    end

    include("test_docs.jl")
end # testset
