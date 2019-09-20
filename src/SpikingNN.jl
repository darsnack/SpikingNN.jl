module SpikingNN

using LightGraphs, MetaGraphs
using DataStructures
using Distributions
using DSP

export  excite!, simulate!, step!, reset!,
		LIF, SRM0,
        George, STDP,
        Population,
        neurons, synapses, inputs, findinputs, outputs, findoutputs, setclass,
        constant_rate, step_current

include("utils.jl")
include("synapse.jl")
include("neuron.jl")
include("models/lif.jl")
include("models/srm0.jl")
include("learning.jl")
include("population.jl")
include("inputs.jl")

end