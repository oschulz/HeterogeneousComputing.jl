# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package HeterogeneousComputing" begin
    # include("test_aqua.jl")
    include("test_compute_unit.jl")
    include("test_docs.jl")
    isempty(Test.detect_ambiguities(HeterogeneousComputing))
end # testset
