"""
    AbstractNeuron

Inherit from this type when creating specific neuron models.

Type Parameters:
- `VT<:Real`: voltage type (also used for current)
- `IT<:Integer`: time stamp index type

Expected Fields:
- `voltage::Real`: membrane potential
- `spikes_in::Queue{Tuple{Integer, Real}}`: a FIFO of input spike times and current at each time stamp
- `last_spike::Integer`: the last time this neuron processed a spike
- `record_fields::Array{Symbol}`: an array of the field names to record
- `record::Dict{Symbol, Array{<:Any}}`: a record of values of symbols in `record_fields`
"""
abstract type AbstractNeuron{VT<:Real, IT<:Integer} end

"""
    record!(neuron::AbstractNeuron, field::Symbol)

Start recording values of `field` in `neuron`.
"""
function record!(neuron::AbstractNeuron, field::Symbol)
    !(field ∈ neuron.record_fields) && push!(neuron.record_fields, field)
    if !haskey(neuron.record, field)
        neuron.record[field] = typeof(getproperty(neuron, field))[]
    end
end

"""
    record!(neurons::Array{AbstractNeuron}, field::Symbol)

Start recording values of `field` for each neuron in `neurons`.
"""
function record!(neurons::Array{AbstractNeuron}, field::Symbol)
    for neuron in neurons
        record!(neuron, field)
    end
end

"""
    derecord!(neuron::AbstractNeuron, field::Symbol)

Stop recording values of `field` in `neuron`.
"""
derecord!(neuron::AbstractNeuron, field::Symbol) = filter!(x -> x != field, neuron.record_fields)

"""
    derecord!(neurons::Array{AbstractNeuron}, field::Symbol)

Stop recording values of `field` for each neuron in `neurons`.
"""
function derecord!(neurons::Array{AbstractNeuron}, field::Symbol)
    for neuron in neurons
        derecord!(neuron, field)
    end
end

"""
    excite!(neuron::AbstractNeuron, spikes::Array{Integer})

Queue an array of spike into a neuron's input queue (w/ weight `= 1.0`).

Fields:
- `neuron::AbstractNeuron`: the neuron to excite
- `spikes::Array{Integer}`: an array of spike times
"""
function excite!(neuron::AbstractNeuron, spikes::Array{<:Integer})
    for t in spikes
        enqueue!(neuron.spikes_in, (t, 1.0))
    end
end

"""
    simulate!(neuron::AbstractNeuron, dt::Real = 1.0)

Fields:
- `neuron::AbstractNeuron`: the neuron to simulate
- `dt::Real`: the simulation time step
"""
function simulate!(neuron::AbstractNeuron, dt::Real = 1.0)
    spike_times = Int[]
    while !isempty(neuron.spikes_in)
        push!(spike_times, step!(neuron, dt))
    end

    filter!(x -> x != 0, spike_times)
end

# """
#     AbstractPopulation{NT<:AbstractNeuron} <: AbstractArray{NT, 1}

# Inherit from this type when creating a population of neurons.

# Parameterized Types:
# - `NT<:AbstractNeuron`: the type of the neurons in the population

# Expected Fields:
# - `graph::AbstractGraph`: the connectivity graph between neurons
# - `neurons::Array{AbstractNeuron}`: an array of neurons in the population
# - `events::Queue{Integer}`: a FIFO of neuron indices indicating spike events
# """
# abstract type AbstractPopulation{NT<:AbstractNeuron} <: AbstractArray{NT, 1} end

# Base.size(pop::AbstractPopulation) = size(neurons)
# Base.IndexStyle(::Type{<:AbstractPopulation}) = IndexLinear()
# Base.getindex(pop::AbstractPopulation, i::Integer) = pop.neurons[i]
# Base.setindex!(pop::AbstractPopulation{NT}, neuron::NT, i::Integer) where {NT<:AbstractNeuron} =
#     (pop.neurons[i] = neuron)