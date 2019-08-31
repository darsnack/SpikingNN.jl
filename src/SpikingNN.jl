module SpikingNN

using LightGraphs, SimpleWeightedGraphs
using DataStructures
using Distributions

export  AbstractNeuron, AbstractPopulation,
        excite!, simulate!, step!, reset!,
		# neurons, synapses, inputs, outputs,
		LIF, SRM0,
        constant_current

include("neuron.jl")
include("lif.jl")
include("srm0.jl")
include("inputs.jl")

end