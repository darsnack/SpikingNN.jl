"""
    LIF

A leaky-integrate-fire neuron.

Fields:
- `voltage::VT`: membrane potential
- `spikes_in::Queue{Tuple{IT, VT}}`: a FIFO of input spike times and current at each time stamp
- `last_spike::IT`: the last time this neuron processed a spike
- `record_fields::Array{Symbol}`: an array of the field names to record
- `record::Dict{Symbol, Array{Any}}`: a record of values of symbols in `record_fields`
- `τ_m::VT`: membrane time constant
- `v_reset::VT`: reset voltage potential
- `v_th::VT`: threshold voltage potential
- `R::VT`: resistive constant (typically = 1)
"""
mutable struct LIF{VT<:Real, IT<:Integer} <: AbstractNeuron{VT, IT}
    # required fields
    voltage::VT
    spikes_in::Queue{Tuple{IT, VT}}
    last_spike::IT
    record_fields::Array{Symbol}
    record::Dict{Symbol, Array{<:Any}}

    # model specific fields
    τ_m::VT
    v_reset::VT
    v_th::VT
    R::VT
end

"""
    LIF(τ_m, v_reset, v_th, R = 1.0)

Create a LIF neuron with zero initial voltage and empty spike queue.
"""
LIF(τ_m::T, v_reset::T, v_th::T, R::T = 1.0) where {T<:Real} =
    LIF{T, Int}(0.0, Queue{Tuple{Int, T}}(), 1, [], Dict{Symbol, Array}(), τ_m, v_reset, v_th, R)


"""
    step!(neuron::LIF, dt::Real = 1.0)::Integer

Evaluate the differential equation between `neuron.last_spike` and the latest input spike.
Return time stamp if the neuron spiked and zero otherwise.
"""
function step!(neuron::LIF{VT<:Real, IT<:Real}, dt::Real = 1.0)
    # pop the latest spike of the queue
    t, current_in = dequeue!(neuron.spikes_in)

    # println("Processing time $(neuron.last_spike) to $t")

    # decay the voltage between last_spike and t
    # println("  v = $(neuron.voltage)")
    for i in neuron.last_spike:t
        neuron.voltage = neuron.voltage - neuron.voltage / neuron.τ_m
        (:voltage ∈ neuron.record_fields && i < t) && push!(neuron.record[:voltage], neuron.voltage)
        # println("  v = $(neuron.voltage)")
    end

    # accumulate the input spike
    neuron.voltage += neuron.R / neuron.τ_m * current_in
    # println("  v (post spike) = $(neuron.voltage)")

    # choose whether to spike
    spiked = (neuron.voltage >= neuron.v_th)
    # println("  spiked? (v_th = $(neuron.v_th)) = $spiked")
    neuron.voltage = spiked ? neuron.v_reset : neuron.voltage
    # println("  v (post thresh) = $(neuron.voltage)")

    # update the last spike
    neuron.last_spike = t + 1

    # record any fields
    for field in neuron.record_fields
        push!(neuron.record[field], getproperty(neuron, field))
    end

    return spiked ? t : 0
end