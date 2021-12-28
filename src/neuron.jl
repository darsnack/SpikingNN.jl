"""
    AbstractCell

Inherit from this type to create a neuron cell (e.g. [`LIF`](@ref)).
"""
abstract type AbstractCell end

function evaluate!(dstate, state, body::AbstractCell, t, current; dt = 1)
    differential!(dstate, state, body, t * dt, current)
    @. state += dstate * dt

    return state
end

refactor!(state, body::AbstractCell, synapses, spikes; dt = 1) = nothing

reset!(state, body::AbstractCell) = nothing

abstract type AbstractThreshold end

"""
    Neuron{BT<:AbstractCell, TT<:AbstractThreshold}

A `Neuron` is a cell body + a threshold.
"""
struct Neuron{BT<:AbstractCell, TT<:AbstractThreshold}
    body::BT
    threshold::TT
end

init(neuron::Neuron) = init(neuron.body)

getvoltage(neuron::Neuron, state) = getvoltage(neuron.body, state)

differential!(dstate, state, neuron::Neuron, t, currents) =
    differential!(dstate, state, neuron.body, t, currents)

"""
    evaluate!(neuron::Neuron, t; dt = 1)

Evaluate the neuron's cell body, decide whether to spike according to the
 threshold, then register the spike event with the cell body.
Return the spike event (0 for no spike or `t` for a spike).
"""
function evaluate!(spikes, dstate, state, neuron::Neuron, t, current; dt = 1)
    evaluate!(dstate, state, neuron.body, t, current; dt = dt)
    voltage = getvoltage(neuron, state)
    evaluate!(spikes, neuron.threshold, t, voltage; dt = dt)

    return spikes
end

refactor!(state, neuron::Neuron, synapses, spikes; dt = 1) =
    refactor!(state, neuron.body, synapses, spikes; dt = dt)

"""
    reset!(neuron::Neuron, state)

Reset `neuron` body.
"""
reset!(state, neuron::Neuron) = reset!(state, neuron.body)
