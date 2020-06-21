using Documenter, SpikingNN

makedocs(;
    modules=[SpikingNN],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/darsnack/SpikingNN.jl/blob/{commit}{path}#L{line}",
    sitename="SpikingNN.jl",
    authors="Kyle Daruwalla",
    assets=String[],
)

deploydocs(;
    repo="github.com/darsnack/SpikingNN.jl",
)