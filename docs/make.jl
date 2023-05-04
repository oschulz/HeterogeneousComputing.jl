# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using HeterogeneousComputing

# Doctest setup
DocMeta.setdocmeta!(
    HeterogeneousComputing,
    :DocTestSetup,
    :(using HeterogeneousComputing);
    recursive=true,
)

makedocs(
    sitename = "HeterogeneousComputing",
    modules = [HeterogeneousComputing],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://oschulz.github.io/HeterogeneousComputing.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    strict = !("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/oschulz/HeterogeneousComputing.jl.git",
    forcepush = true,
    push_preview = true,
)
