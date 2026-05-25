# Standard stuff
cd(@__DIR__)
CI = get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
using CairoMakie, Documenter, Literate
using DocumenterTools: Themes
using DocumenterCitations
ENV["JULIA_DEBUG"] = "Documenter"

using Tartaros

bib = CitationBibliography(
    joinpath(@__DIR__, "src/assets", "tartaros.bib");
    style=:authoryear
)

# Literate.markdown("src/examples/greenland-correction.jl",
#     "src/examples"; credit = false)
Literate.markdown("src/examples/ghf-models.jl",
    "src/examples"; credit = false)


makedocs(;
    modules=[Tartaros],
    authors="Jan Swierczek-Jereczek",
    sitename="Tartaros.jl",
    format = Documenter.HTML(
        prettyurls = CI,
        assets = [
            asset("https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap", class=:css),
        ],
        collapselevel = 2,
        ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            # "Topographic correction" => "examples/greenland-correction.md",
            "Bedrock diffusion models" => "examples/ghf-models.md",
        ],
        "API" => "API_public.md",
        "References" => "references.md",
    ],
    doctest = CI,
    draft = false,
    plugins = [bib],
    checkdocs = :none,
    warnonly = true,
)

deploydocs(;
    repo="github.com/fesmc/Tartaros.jl",
    # devbranch="main",
)
