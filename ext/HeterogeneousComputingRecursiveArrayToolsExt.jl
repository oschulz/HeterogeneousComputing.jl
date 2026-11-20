# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

module HeterogeneousComputingRecursiveArrayToolsExt

import Adapt
using HeterogeneousComputing: DeviceUnit
using RecursiveArrayTools: DiffEqArray

# MLDataDevices moves the state arrays and keeps the original time axis.
Adapt.adapt_structure(unit::DeviceUnit, x::DiffEqArray) = Adapt.adapt_storage(unit, x)

end # module HeterogeneousComputingRecursiveArrayToolsExt
