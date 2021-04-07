module SpikingNN

using Distributions
using RecipesBase
using StructArrays
using CUDA
using Adapt
using TensorCast
using Tullio
using LoopVectorization
using Random
using DataStructures: Queue, enqueue!, dequeue!, empty!

using Reexport

export  AbstractSynapse,
        AbstractThreshold,
        AbstractCell,
        AbstractInput,
        AbstractLearner

export  excite!, simulate!, evaluate!, reset!,
        getvoltage,
        Delta, Alpha, BiExponential,
        Ideal, Poisson,
        LIF, SRM0,
        Neuron,
        STDP,
        Population,
        neurons, synapses,
        step!, update!,
        ConstantRate, StepCurrent, PoissonInput,
        InputPopulation,
        Network, connect!,
        cpu, gpu

include("utils/circulararray.jl")
include("utils/generic.jl")
include("utils/plot.jl")

# prototypes
function excite! end
function refactor! end
function evaluate! end
function reset! end
# function isactive end

include("synapse.jl")
include("models/synapse/delta.jl")
include("models/synapse/alpha.jl")
include("models/synapse/biexp.jl")
include("neuron.jl")
include("models/threshold/ideal.jl")
include("models/threshold/poisson.jl")
include("models/neuron/lif.jl")
include("models/neuron/srm0.jl")
include("inputs.jl")
include("learning.jl")
include("population.jl")
include("network.jl")
# include("gpu.jl")

end