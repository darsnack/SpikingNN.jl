import Pkg

using Plots
using VisualRegressionTests
using SpikingNN
using Test
using Distributions
using HypothesisTests

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

@testset "Visual Regression Tests" begin
    ENV["GKSwstype"] = "100"
    for testfile in testfiles
        include(testfile)
    end
end

@testset "Input Models" begin
    include("input-test.jl")
end