module SpikingNN

using LightGraphs, MetaGraphs
using DataStructures
using Distributions
using DSP
using RecipesBase

export  excite!, simulate!, step!, reset!,
        isdone, isactive,
		LIF, SRM0,
        George, STDP,
        Population,
        neurons, synapses, #inputs, findinputs, outputs, findoutputs, setclass,
        update!,
        Synapse, Threshold,
        ConstantRate, StepCurrent, PoissonInput

include("utils.jl")
include("synapse.jl")
include("threshold.jl")
include("neuron.jl")
include("models/lif.jl")
include("models/srm0.jl")
include("inputs.jl")
include("learning.jl")
include("population.jl")

end