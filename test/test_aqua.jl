# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import HeterogeneousComputing

# Runs after the extension test files, so the loaded extension method
# tables are guarded as well:
Test.@testset "Package ambiguities" begin
    ext_mods = filter(
        !isnothing,
        Any[
            Base.get_extension(HeterogeneousComputing, name) for name in (
                :HeterogeneousComputingCUDAExt, :HeterogeneousComputingAMDGPUExt,
                :HeterogeneousComputingGPUArraysExt, :HeterogeneousComputingJLArraysExt,
                :HeterogeneousComputingKernelAbstractionsExt, :HeterogeneousComputingReactantExt
            )
        ]
    )
    Test.@test isempty(Test.detect_ambiguities(HeterogeneousComputing, ext_mods...))
end # testset

Test.@testset "Aqua tests" begin
    # Slow CI runners need more than the default 30 seconds to shut down the
    # process that checks for persistent tasks:
    Aqua.test_all(
        HeterogeneousComputing,
        ambiguities = true,
        persistent_tasks = (tmax = 180,)
    )
end # testset
