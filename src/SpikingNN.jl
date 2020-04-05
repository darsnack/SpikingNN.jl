module SpikingNN

using LightGraphs
using DataStructures
using Distributions
using DSP
using RecipesBase
using UnPack
using SNNlib
using StructArrays

export  excite!, simulate!, reset!,
        getvoltage,
        # isdone, isactive,
        Neuron,
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
include("cell.jl")
include("neuron.jl")
include("models/lif.jl")
include("models/srm0.jl")
include("inputs.jl")
include("learning.jl")
include("population.jl")

# default isactive is false
# isactive(x, t::Integer) = false

end