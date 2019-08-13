"""
    LIF

A leaky-integrate-fire neuron.

Fields:
- `voltage::Real`: membrane potential
- `class::Symbol`: the class of the neuron (:input, :output, or :none)
- `spikes_in::Queue{Integer}`: a FIFO of input spike times
- `last_spike::Integer`: the last time this neuron processed a spike
- `record_fields::Array{Symbol}`: an array of the field names to record
- `record::Dict{Symbol, Array{Any}}`: a record of values of symbols in `record_fields`
- `τ_m::Real`: membrane time constant
- `v_reset::Real`: reset voltage potential
- `v_th::Real`: threshold voltage potential
- `R::Real`: resistive constant (typically = 1)
"""
mutable struct LIF <: AbstractNeuron
    # required fields
    voltage::Real
    class::Symbol
    spikes_in::Queue{<:Integer}
    last_spike::Integer
    record_fields::Array{Symbol}
    record::Dict{Symbol, Array{<:Any}}

    # model specific fields
    τ_m::Real
    v_reset::Real
    v_th::Real
    R::Real
end

"""
    LIF(class, τ_m, v_reset, v_th, R = 1.0)

Create a LIF neuron with zero initial voltage and empty spike queue.
"""
LIF(class::Symbol, τ_m::Real, v_reset::Real, v_th::Real, R::Real = 1.0) =
    LIF(0.0, class, Queue{Int}(), 1, [], Dict{Symbol, Array}(), τ_m, v_reset, v_th, R)


"""
    step!(neuron::LIF, dt::Real = 1.0)::Bool

Evaluate the differential equation between `neuron.last_spike` and the latest input spike.
Return time stamp if the neuron spiked and zero otherwise.
"""
function step!(neuron::LIF, dt::Real = 1.0)
    # pop the latest spike of the queue
    t = dequeue!(neuron.spikes_in)

    # println("Processing time $(neuron.last_spike) to $t")

    # decay the voltage between last_spike and t
    # println("  v = $(neuron.voltage)")
    for i in neuron.last_spike:t
        neuron.voltage = neuron.voltage - neuron.voltage / neuron.τ_m
        (:voltage ∈ neuron.record_fields && i < t) && push!(neuron.record[:voltage], neuron.voltage)
        # println("  v = $(neuron.voltage)")
    end

    # accumulate the input spike
    neuron.voltage += neuron.R / neuron.τ_m
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