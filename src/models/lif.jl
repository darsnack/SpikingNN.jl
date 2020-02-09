"""
    LIF

A leaky-integrate-fire neuron.

Fields:
- `voltage::VT`: membrane potential
- `current_in::Accumulator{IT, VT}`: a map of time index => current at each time stamp
- `last_spike::IT`: the last time this neuron processed a spike
- `τ_m::VT`: membrane time constant
- `v_reset::VT`: reset voltage potential
- `v_th::VT`: threshold voltage potential
- `R::VT`: resistive constant (typically = 1)
"""
mutable struct LIF{VT<:Real, IT<:Integer} <: AbstractNeuron{VT, IT}
    # required fields
    voltage::VT
    current_in::Accumulator{IT, VT}

    # model specific fields
    last_spike::IT
    τ_m::VT
    v_reset::VT
    v_th::VT
    R::VT
end

Base.show(io::IO, ::MIME"text/plain", neuron::LIF) =
    print(io, """LIF with $(length(neuron.current_in)) queued current events:
                     voltage: $(neuron.voltage)
                     τ_m:     $(neuron.τ_m)
                     v_reset: $(neuron.v_reset)
                     v_th:    $(neuron.v_th)
                     R:       $(neuron.R)""")
Base.show(io::IO, neuron::LIF) =
    print(io, "LIF(τ_m: $(neuron.τ_m), v_reset: $(neuron.v_reset), v_th: $(neuron.v_th), R: $(neuron.R))")

"""
    LIF(τ_m, v_reset, v_th, R = 1.0)

Create a LIF neuron with zero initial voltage and empty current queue.
"""
LIF(τ_m::Real, v_reset::Real, v_th::Real, R::Real = 1.0) =
    LIF{Float64, Int}(v_reset, Accumulator{Int, Float64}(), 1, τ_m, v_reset, v_th, R)

"""
    isactive(neuron::LIF, t::Integer)

Return true if the neuron has a current event to process at this time step `t`.
"""
isactive(neuron::LIF, t::Integer) = haskey(neuron.current_in, t)

"""
    (neuron::LIF)(t::Integer; dt::Real = 1.0)

Evaluate the neuron model at time `t`.
Return time stamp if the neuron spiked and zero otherwise.
"""
function (neuron::LIF)(t::Integer; dt::Real = 1.0)
    # pop the latest spike off the queue
    current_in = DataStructures.reset!(neuron.current_in, t)

    # println("Processing time $(neuron.last_spike) to $t")

    # decay the voltage between last_spike and t
    # println("  v = $(neuron.voltage)")
    for i in neuron.last_spike:t
        neuron.voltage = neuron.voltage - neuron.voltage / neuron.τ_m
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

    return spiked ? t : 0
end

"""
    reset!(neuron::LIF)

Reset the neuron to its reset voltage and clear its input current queue.
"""
function reset!(neuron::LIF)
    neuron.voltage = neuron.v_reset
    neuron.last_spike = 1
    for key in keys(neuron.current_in)
        reset!(neuron.current_in, key)
    end
end