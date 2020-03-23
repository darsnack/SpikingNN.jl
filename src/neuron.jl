using .Synapse: sampleresponse, AbstractSynapse

"""
    AbstractNeuron

Inherit from this type when creating specific neuron models.

Type Parameters:
- `VT<:Real`: voltage type (also used for current)
- `IT<:Integer`: time stamp index type

Expected Fields:
- `voltage::VT`: membrane potential
- `current_in::Accumulator{IT, VT}`: a map of time index => current at each time stamp
"""
abstract type AbstractNeuron{VT<:Real, IT<:Integer} end

"""
    isdone(neuron::AbstractNeuron)

Return true if the neuron has no more current events to process.
"""
isdone(neuron::AbstractNeuron) = isempty(neuron.current_in)

"""
    excite!(neuron::AbstractNeuron, spikes::Array{Integer})

Excite a neuron with spikes according to a synaptic function.

Fields:
- `neuron::AbstractNeuron`: the neuron to excite
- `spikes::Array{Integer}`: an array of spike times
- `synapse::AbstractSynapse`: a synaptic function applied to each spike
- `weight::Real`: a weight applied to excitation current
- `dt::Real`: the sample rate for the response function
"""
function excite!(neuron::AbstractNeuron, spikes::Array{<:Integer};
                 dt::Real = 1.0, synapse::AbstractSynapse = Synapse.Delta(dt = dt), weight::Real = 1)
    # push spikes onto synapse
    push!(synapse, spikes)

    # get currents from synapse
    T = maximum(spikes)
    n = Int(ceil(T / dt))
    current = [synapse(t; dt = dt) for t in 1:n]

    # increment current
    @inbounds for t in 1:n
        inc!(neuron.current_in, )
end

"""
    excite!(neuron::AbstractNeuron, spike::Integer)

Excite a neuron with spikes according to a response function.
Faster by not using convolution for single spike.

Fields:
- `neuron::AbstractNeuron`: the neuron to excite
- `spike::Integer`: spike time
- `response::AbstractSynapse`: a response function applied to each spike
- `weight::Real`: a weight applied to excitation current
- `dt::Real`: the sample rate for the response function
"""
function excite!(neuron::AbstractNeuron, spike::Integer;
                 dt::Real = 1.0, response::AbstractSynapse = Synapse.Delta(dt = dt), weight::Real = 1)
    # sample the response function
    h, N = sampleresponse(response)
    h = weight .* h

    # increment current
    @inbounds for t in 1:N
        inc!(neuron.current_in, spike + t - 1, h[t])
    end
end

"""
    excite!(neuron::AbstractNeuron, input, T::Integer)

Excite a neuron with spikes from an input function according to a response function.
Return the spike time array due to input function.

Fields:
- `neuron::AbstractNeuron`: the neuron to excite
- `input::(t::Integer; dt::Real) -> {0, 1}`: an input function to excite the neuron
- `response::AbstractSynapse`: a response function applied to each spike
- `weight::Real`: a weight applied to excitation current
- `dt::Real`: the sample rate for the response function
"""
function excite!(neuron::AbstractNeuron, input, T::Integer;
                 dt::Real = 1.0, response::AbstractSynapse = Synapse.Delta(dt = dt), weight::Real = 1)
    spike_times = filter!(x -> x != 0, [input(t; dt = dt) for t = 1:T])
    excite!(neuron, spike_times; response = response, dt = dt, weight = weight)

    return spike_times
end

"""
    excite!(neurons::Array{AbstractNeuron}, spikes::Array{Integer})

Excite an array of neurons with spikes according to a response function.

Fields:
- `neurons::Array{AbstractNeuron}`: an array of neurons to excite
- `spikes::Array{Integer}`: an array of spike times
- `response::AbstractSynapse`: a response function applied to each spike
- `weight::Real`: a weight applied to excitation current
- `dt::Real`: the sample rate for the response function
"""
function excite!(neurons::Array{<:AbstractNeuron}, spikes::Array{<:Integer};
                 dt::Real = 1.0, response::AbstractSynapse = Synapse.Delta(dt = dt), weight::Real = 1)
    for neuron in neurons
        excite!(neurons, spikes; response = response, dt = dt, weight = weight)
    end
end

"""
    excite!(neurons::Array{AbstractNeuron}, spikes::Integer)

Excite an array of neurons with spikes according to a response function.
Faster by not using convolution for single spike.

Fields:
- `neurons::Array{AbstractNeuron}`: an array of neurons to excite
- `spikes::Integer`: spike time
- `response::AbstractSynapse`: a response function applied to each spike
- `weight::Real`: a weight applied to excitation current
- `dt::Real`: the sample rate for the response function
"""
function excite!(neurons::Array{<:AbstractNeuron}, spike::Integer;
                 dt::Real = 1.0, response::AbstractSynapse = Synapse.Delta(dt = dt), weight::Real = 1)
    for neuron in neurons
        excite!(neuron, spike; response = response, dt = dt, weight = weight)
    end
end

"""
    excite!(neurons::Array{AbstractNeuron}, input, T::Integer)

Excite an array of neurons with spikes from an input function according to a response function.
Return the spike time array due to input function.

Fields:
- `neurons::Array{AbstractNeuron}`: an array of neurons to excite
- `input::(t::Integer; dt::Real) -> {0, 1}`: an input function to excite the neuron
- `response::AbstractSynapse`: a response function applied to each spike
- `weight::Real`: a weight applied to excitation current
- `dt::Real`: the sample rate for the response function
"""
function excite!(neurons::Array{<:AbstractNeuron}, input, T::Integer;
                 dt::Real = 1.0, response::AbstractSynapse = Synapse.Delta(dt = dt), weight::Real = 1)
    spike_times = Array{Int}[]

    for (i, neuron) in enumerate(neurons)
        push!(spike_times[i], excite!(neuron, input, T; response = response, dt = dt, weight = weight))
    end

    return spike_times
end

"""
    simulate!(neuron::AbstractNeuron)

Fields:
- `neuron::AbstractNeuron`: the neuron to simulate
- `T::Integer`: number of time steps to simulate
- `dt::Real`: the length ofsimulation time step
- `cb::Function`: a callback function that is called after event evaluation
- `dense::Bool`: set to `true` to evaluate every time step even in the absence of events
"""
function simulate!(neuron::AbstractNeuron, T::Integer; dt::Real = 1.0, cb = () -> (), dense = false)
    spike_times = Int[]

    # step! neuron until queue is empty
    cb()
    for t = 1:T
        if isactive(neuron, t) || dense
            push!(spike_times, neuron(t; dt = dt))
            cb()
        end
    end

    return filter!(x -> x != 0, spike_times)
end