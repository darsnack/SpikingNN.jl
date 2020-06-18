using Pkg

using Plots
using VisualRegressionTests
using SpikingNN
using Test

# environment settings
istravis = "TRAVIS" ∈ keys(ENV)
datadir = joinpath(@__DIR__, "data")
if !istravis
    Pkg.add("Gtk")
    using Gtk
end

# test files to include
testfiles = [
    "excite-test.jl",
    "lif-test.jl",
    "population-test.jl",
    "srm0-test.jl",
    "srm0-threshold-test.jl",
    "stdp-test.jl"
]

@testset "SpikingNN.jl" begin
    for testfile in testfiles
        include(testfile)
    end
end