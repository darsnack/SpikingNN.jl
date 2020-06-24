using Documenter, SpikingNN

makedocs(;
    modules=[SpikingNN],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Usage" => [
            "Neurons" => "neuron.md",
            "Populations" => "population.md",
            "Inputs" => "input.md"
            "Networks" => "network.md"
        ],
        "Models" => [
            "Neuron Models" => "neuron-models.md",
            "Synapse Models" => "synapse-models.md",
            "Threshold Models" => "threshold-models.md"
        ],
    ],
    repo="https://github.com/darsnack/SpikingNN.jl/blob/{commit}{path}#L{line}",
    sitename="SpikingNN.jl",
    authors="Kyle Daruwalla",
    assets=String[],
)

deploydocs(;
    repo="github.com/darsnack/SpikingNN.jl",
    versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
)