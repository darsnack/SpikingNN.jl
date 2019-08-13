module SpikingNN

using LightGraphs, SimpleWeightedGraphs
using DataStructures
using Distributions

export  AbstractNeuron, AbstractPopulation,
        excite!, simulate!, step!,
		neurons, synapses, inputs, outputs,
		LIF,
        constant_current

include("neuron.jl")
include("lif.jl")
include("inputs.jl")

end