# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

"""
    HeterogeneousComputing

Tools for heterogeneous computing in Julia.
"""
module HeterogeneousComputing

using Random

import Adapt

include("compute_unit.jl")
include("gen_context.jl")

@static if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/HeterogeneousComputingCUDAExt.jl")
        @require KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c" include("../ext/HeterogeneousComputingKernelAbstractionsExt.jl")
    end
end

end # module
