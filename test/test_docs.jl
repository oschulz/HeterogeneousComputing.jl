# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

using Test
using HeterogeneousComputing
import Documenter

Documenter.DocMeta.setdocmeta!(
    HeterogeneousComputing,
    :DocTestSetup,
    :(using HeterogeneousComputing);
    recursive = true
)
Documenter.doctest(HeterogeneousComputing)
