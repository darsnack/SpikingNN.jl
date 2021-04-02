"""
    AbstractCell

Inherit from this type to create a neuron cell (e.g. [`LIF`](@ref)).
"""
abstract type AbstractCell end

abstract type AbstractThreshold end


"""
    Neuron{BT<:AbstractCell, TT<:AbstractThreshold}

A `Neuron` is a cell body + a threshold.
"""
struct Neuron{BT<:AbstractCell, TT<:AbstractThreshold}
    body::BT
    threshold::TT
end

getvoltage(neuron::Neuron) = getvoltage(neuron.body)
isactive(neuron::Neuron, t::Integer; dt::Real = 1.0) =
    isactive(neuron.body, t; dt = dt) || isactive(neuron.threshold, t; dt = dt)

"""
    evaluate!(neuron::Neuron, t::Integer; dt::Real = 1.0)
    (::Neuron)(t::Integer; dt::Real = 1.0)
    evaluate!(somas::AbstractArray{<:Neuron}, t::Integer; dt::Real = 1.0)

Evaluate the neuron's cell body, decide whether to spike according to the
 threshold, then register the spike event with the cell body.
Return the spike event (0 for no spike or `t` for a spike).
"""
function evaluate!(neuron::Neuron, t::Integer, current; dt::Real = 1.0)
    voltage = evaluate!(neuron.body, t, current; dt = dt)
    spike = evaluate!(neuron.threshold, t, voltage; dt = dt)
    # spike!(neuron.body, spike; dt = dt)

    return spike
end
(neuron::Neuron)(t::Integer, current; dt::Real = 1.0) = evaluate!(neuron, t; dt = dt)
function evaluate!(neurons::T, t::Integer, currents; dt::Real = 1.0) where T<:AbstractArray{<:Neuron}
    voltage = evaluate!(neurons.body, t, currents; dt = dt)
    spikes = evaluate!(neurons.threshold, t, voltage; dt = dt)
    # spike!(neurons.body, spikes; dt = dt)

    return spikes
end
function evaluate!(spikes, neurons::T, t::Integer, currents; dt::Real = 1.0) where T<:AbstractArray{<:Neuron}
    voltage = evaluate!(neurons.body, t, currents; dt = dt)
    evaluate!(spikes, neurons.threshold, t, voltage; dt = dt)
    # spike!(neurons.body, spikes; dt = dt)

    return spikes
end

"""
    reset!(neuron::T) where T<:Union{Soma, AbstractArray{<:Soma}}

Reset `neuron.body`.
"""
reset!(neuron::Neuron) = reset!(neuron.body)
reset!(neurons::AbstractArray{<:Neuron}) = reset!(neurons.body)
