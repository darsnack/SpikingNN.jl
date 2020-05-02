abstract type AbstractCell end

struct Soma{BT<:AbstractCell, TT<:AbstractThreshold}
    body::BT
    threshold::TT
end

getvoltage(soma::Soma) = getvoltage(soma.body)
isactive(soma::Soma, t::Integer; dt::Real = 1.0) = isactive(soma.body, t; dt = dt) || isactive(soma.threshold, t; dt = dt)
excite!(soma::T, current) where T<:Union{Soma, AbstractArray{<:Soma}} = excite!(soma.body, current)

function (soma::Soma)(t::Integer; dt::Real = 1.0)
    spike = soma.threshold(t, soma.body(t; dt = dt); dt = dt)
    spike!(soma.body, spike; dt = dt)

    return spike
end
function evalsomas(somas::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Soma}
    voltage = evalcells(somas.body, t; dt = dt)
    spikes = evalthresholds(somas.threshold, t, voltage; dt = dt)
    spike!(somas.body, spikes; dt = dt)

    return spikes
end

reset!(soma::T) where T<:Union{Soma, AbstractArray{<:Soma}} = reset!(soma.body)

struct Neuron{ST<:AbstractArray{<:AbstractSynapse}, CT<:Soma}
    synapses::ST
    soma::CT
end
Neuron(synapse::ST, body::BT, threshold::TT) where {ST<:AbstractSynapse, BT<:AbstractCell, TT<:AbstractThreshold} =
    Neuron(StructArray([synapse]; unwrap = t -> t <:AbstractSynapse), Soma(body, threshold))
Neuron{ST}(body::BT, threshold::TT) where {ST<:AbstractSynapse, BT<:AbstractCell, TT<:AbstractThreshold} =
    Neuron(StructArray{ST}(undef, 0), Soma(body, threshold))

function connect!(neuron::Neuron, synapse::AbstractSynapse)
    push!(neuron.synapses, synapse)

    return neuron
end

getvoltage(neuron::Neuron) = getvoltage(neuron.soma)
isactive(neuron::Neuron, t::Integer; dt::Real = 1.0) = isactive(neuron.soma, t; dt = dt) || isactive(neuron.synapses, t; dt = dt)

excite!(neuron::Neuron, spike::Integer) = excite!(neuron.synapses, spike)
excite!(neuron::Neuron, spikes::Array{<:Integer}) = excite!(neuron.synapses, spikes)
function excite!(neuron::Neuron, input, T::Integer; dt::Real = 1.0)
    spikes = filter!(x -> x != 0, [input(t; dt = dt) for t = 1:T])
    excite!(neuron, spikes)

    return spikes
end
function excite!(neuron::Neuron, inputs::AbstractVector, T::Integer; dt::Real = 1.0)
    spikearray = Array[]
    for (i, input) in enumerate(inputs)
        spikes = filter!(x -> x != 0, [input(t; dt = dt) for t = 1:T])
        excite!(view(neuron.synapses, i), spikes)
        push!(spikearray, spikes)
    end

    return spikearray
end

function (neuron::Neuron)(t::Integer; dt::Real = 1.0)
    I = sum(evalsynapses(neuron.synapses, t; dt = dt))
    excite!(neuron.soma, I)
    spike = neuron.soma(t; dt = dt)

    return spike
end

function reset!(neuron::T) where T<:Union{Neuron, AbstractArray{<:Neuron}}
    reset!(neuron.synapses)
    reset!(neuron.soma)
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
    for t = 1:T
        if dense || isactive(neuron, t; dt = dt)
            cb()
            push!(spikes, neuron(t; dt = dt))
        end
    end

    return filter!(x -> x != 0, spikes)
end