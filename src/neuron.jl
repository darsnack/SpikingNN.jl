"""
    AbstractCell

Inherit from this type to create a neuron cell (e.g. [`LIF`](@ref)).
"""
abstract type AbstractCell end


"""
    Soma{BT<:AbstractCell, TT<:AbstractThreshold}

A `Soma` is a cell body + a threshold.
"""
struct Soma{BT<:AbstractCell, TT<:AbstractThreshold}
    body::BT
    threshold::TT
end

"""
    getvoltage(soma::Soma)

Get the current membrane potential of `soma.body`.
"""
getvoltage(soma::Soma) = getvoltage(soma.body)
isactive(soma::Soma, t::Integer; dt::Real = 1.0) = isactive(soma.body, t; dt = dt) || isactive(soma.threshold, t; dt = dt)

"""
    excite!(soma, current)

Inject `current` into `soma.body`.
"""
excite!(soma::T, current) where T<:Union{Soma, AbstractArray{<:Soma}} = excite!(soma.body, current)

"""
    (::Soma)(t::Integer; dt::Real = 1.0)
    evalsomas(somas::AbstractArray{<:Soma}, t::Integer; dt::Real = 1.0)

Evaluate the soma's cell body, decide whether to spike according to the
 threshold, then register the spike event with the cell body.
Return the spike event (0 for no spike or `t` for a spike).
"""
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


"""
    Neuron{ST<:AbstractArray{<:AbstractSynapse}, CT<:Soma}

A `Neuron` is a vector of synapses feeding into a soma.
"""
struct Neuron{ST<:AbstractArray{<:AbstractSynapse}, CT<:Soma}
    synapses::ST
    soma::CT
end
Neuron(synapse::ST, body::BT, threshold::TT) where {ST<:AbstractSynapse, BT<:AbstractCell, TT<:AbstractThreshold} =
    Neuron(StructArray([synapse]; unwrap = t -> t <:AbstractSynapse), Soma(body, threshold))
Neuron{ST}(body::BT, threshold::TT) where {ST<:AbstractSynapse, BT<:AbstractCell, TT<:AbstractThreshold} =
    Neuron(StructArray{ST}(undef, 0), Soma(body, threshold))

"""
    connect!(neuron::Neuron, synapse::AbstractSynapse)

Connect `synapse` into `neuron`.
"""
function connect!(neuron::Neuron, synapse::AbstractSynapse)
    push!(neuron.synapses, synapse)

    return neuron
end

"""
    getvoltage(neuron::Neuron)

Get the current membrane potential of `neuron.soma`.
"""
getvoltage(neuron::Neuron) = getvoltage(neuron.soma)
isactive(neuron::Neuron, t::Integer; dt::Real = 1.0) = isactive(neuron.soma, t; dt = dt) || isactive(neuron.synapses, t; dt = dt)

"""
    excite!(neuron::Neuron, spike::Integer)
    excite!(neuron::Neuron, spikes::Array{<:Integer})
    excite!(neuron::Neuron, input, T::Integer; dt::Real = 1.0)

Excite a `neuron`'s synapses with spikes.
Or excite a `neuron`'s synapses with an arbitrary `input` function evaluated from `1:T`.
"""
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

"""
    (::Neuron)(t::Integer; dt::Real = 1.0)

Evaluate a neuron at time `t` by evaluating all its synapses,
 exciting the soma with current, then registering post-synaptic
 spikes with the synapses.
Return the spike event (0 for no spike or `t` for spike).
"""
function (neuron::Neuron)(t::Integer; dt::Real = 1.0)
    I = sum(evalsynapses(neuron.synapses, t; dt = dt))
    excite!(neuron.soma, I)
    spike = neuron.soma(t; dt = dt)
    spike!(neuron.synapses, spike; dt = dt)

    return spike
end

function reset!(neuron::T) where T<:Union{Neuron, AbstractArray{<:Neuron}}
    reset!(neuron.synapses)
    reset!(neuron.soma)
end

"""
    simulate!(neuron::AbstractNeuron, T::Integer; dt::Real = 1.0, cb = () -> (), dense = false)

Fields:
- `neuron::AbstractNeuron`: the neuron to simulate
- `T::Integer`: number of time steps to simulate
- `dt::Real`: the length of simulation time step
- `cb::Function`: a callback function that is called at the start of each time step
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