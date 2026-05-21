using Tartaros
using Documenter

DocMeta.setdocmeta!(Tartaros, :DocTestSetup, :(using Tartaros); recursive=true)

makedocs(;
    modules=[Tartaros],
    authors="Jan Swierczek-Jereczek",
    sitename="Tartaros.jl",
    format=Documenter.HTML(;
        canonical="https://fesmc.github.io/Tartaros.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/fesmc/Tartaros.jl",
    devbranch="main",
)
