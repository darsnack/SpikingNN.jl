module SpikingNN

using Distributions
using RecipesBase
using StructArrays
using CUDA
using Adapt
using TensorCast
using Random

using Reexport

export  AbstractSynapse,
        AbstractThreshold,
        AbstractCell,
        AbstractInput,
        AbstractLearner

export  excite!, simulate!, evaluate!, reset!,
        getvoltage,
        Soma, Neuron,
        LIF, SRM0,
        George, STDP,
        Population,
        neurons, synapses,
        prespike!, postspike!, record!, update!,
        ConstantRate, StepCurrent, PoissonInput,
        InputPopulation,
        Network, connect!,
        cpu, gpu

include("utils.jl")

# prototypes
function excite! end
function spike! end
function evaluate! end
function reset! end
function isactive end

include("synapse.jl")
include("threshold.jl")
include("neuron.jl")
include("models/lif.jl")
include("models/srm0.jl")
include("inputs.jl")
include("learning.jl")
include("population.jl")
include("network.jl")
include("gpu.jl")

end