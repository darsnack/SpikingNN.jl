"""
    AbstractNeuron

Inherit from this type when creating specific neuron models.

Type Parameters:
- `VT<:Real`: voltage type (also used for current)
- `IT<:Integer`: time stamp index type

Expected Fields:
- `voltage::VT`: membrane potential
- `spikes_in::Accumulator{IT, VT}`: a map of input spike times => post-synaptic potential at that time
- `last_spike::IT`: the last time this neuron processed a spike
"""
abstract type AbstractNeuron{VT<:Real, IT<:Integer} end

"""
    excite!(neuron::AbstractNeuron, spikes::Array{Integer})

Queue an array of spike into a neuron's input queue (w/ weight `= 1.0`).

Fields:
- `neuron::AbstractNeuron`: the neuron to excite
- `spikes::Array{Integer}`: an array of spike times
"""
function excite!(neuron::AbstractNeuron, spikes::Array{<:Integer})
    for t in spikes
        inc!(neuron.spikes_in, t)
    end
end

"""
    simulate!(neuron::AbstractNeuron, dt::Real = 1.0)

Fields:
- `neuron::AbstractNeuron`: the neuron to simulate
- `dt::Real`: the simulation time step
- `cb::Function`: a callback function that is called after event evaluation
- `dense::Bool`: set to `true` to evaluate every time step even in the absence of events
"""
function simulate!(neuron::AbstractNeuron, dt::Real = 1.0; cb = () -> (), dense = false)
    spike_times = Int[]

    # for dense evaluation, add spikes with zero current to the queue
    if dense && !isempty(neuron.spikes_in)
        max_t = maximum(keys(neuron.spikes_in))
        for t in setdiff(1:max_t, keys(neuron.spikes_in))
            inc!(neuron.spikes_in, t, 0)
        end
    end

    # step! neuron until queue is empty
    cb()
    while !isempty(neuron.spikes_in)
        push!(spike_times, step!(neuron, dt))
        cb()
    end

    filter!(x -> x != 0, spike_times)
end