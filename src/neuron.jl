abstract type AbstractCell end

struct Neuron{ST<:AbstractArray{<:AbstractSynapse}, BT<:AbstractCell, TT<:AbstractThreshold}
    synapses::ST
    body::BT
    threshold::TT
end
Neuron(synapse::ST, body::BT, threshold::TT) where {ST<:AbstractSynapse, BT<:AbstractCell, TT<:AbstractThreshold} =
    Neuron(StructArray([synapse]), body, threshold)
Neuron{ST}(body::BT, threshold::TT) where {ST<:AbstractSynapse, BT<:AbstractCell, TT<:AbstractThreshold} =
    Neuron(StructArray{ST}(undef, 0), body, threshold)

function connect!(neuron::Neuron, synapse::AbstractSynapse)
    push!(neuron.synapses, synapse)

    return neuron
end

getvoltage(neuron::Neuron) = getvoltage(neuron.body)
isactive(neuron::Neuron, t::Integer; dt::Real = 1.0) = isactive(neuron.body, t; dt = dt) ||
                                                       isactive(neuron.threshold, t; dt = dt) ||
                                                       any(s -> isactive(s, t; dt = dt), neuron.synapses)

excite!(neuron::Neuron, spike::Integer) = map(s -> excite!(s, spike), neuron.synapses)
excite!(neuron::Neuron, spikes::Array{<:Integer}) = map(s -> excite!(s, spikes), neuron.synapses)
function excite!(neuron::Neuron, input, T::Integer; dt::Real = 1.0)
    spikes = filter!(x -> x != 0, [input(t; dt = dt) for t = 1:T])
    excite!(neuron, spikes)

    return spikes
end

function (neuron::Neuron)(t::Integer; dt::Real = 1.0)
    I = sum(evalsynapses(neuron.synapses, t; dt = dt))
    excite!(neuron.body, I)
    spike = neuron.threshold(t, neuron.body(t; dt = dt); dt = dt)
    (spike > 0) && spike!(neuron.body, t; dt = dt)

    return spike
end

function reset!(neuron::Neuron)
    reset!(neuron.synapses)
    reset!(neuron.body)
end
function reset!(neurons::T) where T<:AbstractArray{<:Neuron}
    reset!(neurons.synapses)
    reset!(neurons.body)
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
function simulate!(neuron::Neuron, T::Integer; dt::Real = 1.0, cb = () -> (), dense = false)
    spikes = Int[]

    # step! neuron until queue is empty
    cb()
    for t = 1:T
        if dense || isactive(neuron, t; dt = dt)
            push!(spikes, neuron(t; dt = dt))
            cb()
        end
    end

    return filter!(x -> x != 0, spikes)
end