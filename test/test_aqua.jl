# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import HeterogeneousComputing

Test.@testset "Aqua tests" begin
    Aqua.test_all(HeterogeneousComputing)
end # testset
