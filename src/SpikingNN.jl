module SpikingNN

using Adapt
using CUDA
using LoopVectorization
using StructArrays
using ComponentArrays
using Random
using RecipesBase

using Reexport

export  AbstractSynapse,
        AbstractThreshold,
        AbstractCell,
        AbstractInput,
        AbstractLearner

export  excite!, simulate!, evaluate!, refactor!, reset!,
        getvoltage,
        Delta, Alpha, BiExponential,
        Ideal, Poisson, Probabilistic,
        LIF, SRM0,
        Neuron,
        # STDP,
        # Population,
        # neurons, synapses,
        # step!, update!,
        ConstantRate, StepCurrent, PoissonInput, FunctionalInput#,
        # InputPopulation,
        # Network, connect!,
        # cpu, gpu

include("utils/impulsebuffer.jl")
include("utils/generic.jl")
include("utils/plot.jl")

include("synapse.jl")
include("models/synapse/delta.jl")
include("models/synapse/alpha.jl")
include("models/synapse/biexp.jl")
include("neuron.jl")
include("models/threshold/ideal.jl")
include("models/threshold/poisson.jl")
include("models/threshold/probabilistic.jl")
include("models/neuron/lif.jl")
include("models/neuron/srm0.jl")
include("inputs.jl")
# include("learning.jl")
# include("population.jl")
# include("network.jl")
# include("gpu.jl")

end