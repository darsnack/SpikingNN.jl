"""
    AbstractNeuron

Inherit from this type when creating specific neuron models.

# Expected Fields:
- `voltage::Real`: membrane potential (mV)
- `spikes_in::Array{Integer}`: an array of indices of input spike times
- `spikes_out::Array{Integer}`: an array of indices of output spike times
"""
abstract type AbstractNeuron end

"""
    AbstractPopulation{NT<:AbstractNeuron}

Inherit from this type when creating a population of neurons.

# Parameterized Types:
- `NT<:AbstractNeuron`: the type of the neurons in the population

# Expected Fields:
- `graph::AbstractGraph`: the connectivity graph between neurons
"""
abstract type AbstractPopulation{NT<:AbstractNeuron} end