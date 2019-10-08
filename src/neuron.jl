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

_isactive(neuron::AbstractNeuron, t::Integer) = haskey(neuron.current_in, t)

"""
    isdone(neuron::AbstractNeuron)

Return true if the neuron has no more current events to process.
"""
isdone(neuron::AbstractNeuron) = isempty(neuron.current_in)

"""
    excite!(neuron::AbstractNeuron, spikes::Array{Integer})

Excite a neuron with spikes according to a response function.

Fields:
- `neuron::AbstractNeuron`: the neuron to excite
- `spikes::Array{Integer}`: an array of spike times
- `response::Function`: a response function applied to each spike
- `dt::Real`: the sample rate for the response function
"""
function excite!(neuron::AbstractNeuron, spikes::Array{<:Integer}; response = delta, dt::Real = 1.0)
    # sample the response function
    h, N = sample_response(response, dt)

    # construct a dense version of the spike train
    n = maximum(spikes)
    if (n < N)
        @warn "The number of samples of the response function (N = $N) is larger than the total input length (maximum(spikes) = $n)"
    end
    x = zeros(n)
    x[spikes] .= 1

    # convolve the the response with the spike train
    y = conv(x, h)

    for (t, current) in enumerate(y[1:n])
        (current > 1e-10) && inc!(neuron.current_in, t, current)
    end

    # hack to make sure last spike is in queue even if current is zero
    # (keeps maximum(spikes) consistent even after convolving response)
    inc!(neuron.current_in, n, 0)
end

"""
    excite!(neuron::AbstractNeuron, spikes::Integer)

Excite a neuron with spikes according to a response function.
Faster by not using convolution for single spike.

Fields:
- `neuron::AbstractNeuron`: the neuron to excite
- `spikes::Integer`: spike time
- `response::Function`: a response function applied to each spike
- `dt::Real`: the sample rate for the response function
"""
function excite!(neuron::AbstractNeuron, spike::Integer; response = delta, dt::Real = 1.0)
    # sample the response function
    h, N = sample_response(response, dt)

    # increment current
    for (t, current) in enumerate(h)
        inc!(neuron.current_in, spike + t - 1, current)
    end
end

"""
    excite!(neuron::AbstractNeuron, input, T::Integer)

Excite a neuron with spikes from an input function according to a response function.
Return the spike time array due to input function.

Fields:
- `neuron::AbstractNeuron`: the neuron to excite
- `input::(t::Integer; dt::Real) -> {0, 1}`: an input function to excite the neuron
- `response::Function`: a response function applied to each spike
- `dt::Real`: the sample rate for the response function
"""
function excite!(neuron::AbstractNeuron, input, T::Integer; response = delta, dt::Real = 1.0)
    spike_times = filter!(x -> x != 0, [input(t; dt = dt) ? t : zero(t) for t = 1:T])
    excite!(neuron, spike_times; response = response, dt = dt)

    return spike_times
end

"""
    simulate!(neuron::AbstractNeuron)

Fields:
- `neuron::AbstractNeuron`: the neuron to simulate
- `dt::Real`: the length ofsimulation time step
- `cb::Function`: a callback function that is called after event evaluation
- `dense::Bool`: set to `true` to evaluate every time step even in the absence of events
"""
function simulate!(neuron::AbstractNeuron; dt::Real = 1.0, cb = () -> (), dense = false)
    spike_times = Int[]

    # for dense evaluation, add spikes with zero current to the queue
    if dense && !isempty(neuron.current_in)
        max_t = maximum(keys(neuron.current_in))
        for t in setdiff(1:max_t, keys(neuron.current_in))
            inc!(neuron.current_in, t, 0)
        end
    end

    # step! neuron until queue is empty
    cb()
    t = 1
    while !isempty(neuron.current_in)
        push!(spike_times, neuron(t; dt = dt))
        cb()
        t += 1
    end

    return filter!(x -> x != 0, spike_times)
end