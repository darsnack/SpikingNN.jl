import Pkg

using Plots
using VisualRegressionTests
using SpikingNN
using Test

# environment settings
isci = "CI" ∈ keys(ENV)
datadir = joinpath(@__DIR__, "data")
if !isci
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

@testset "VisualRegressionTests.jl" begin
    ENV["GKSwstype"] = "100"
    for testfile in testfiles
        include(testfile)
    end
end

@testset "inputs.jl" begin
    include("input-test.jl")
end