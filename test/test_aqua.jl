# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import HeterogeneousComputing

Test.@testset "Aqua tests" begin
    # Slow CI runners need more than the default 30 seconds to shut down the
    # process that checks for persistent tasks:
    Aqua.test_all(
        HeterogeneousComputing,
        ambiguities = true,
        persistent_tasks = (tmax = 180,)
    )
end # testset
