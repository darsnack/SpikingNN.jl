"""
    SRM0

A SRM0 neuron.

Fields:
- `voltage::VT`: membrane potential
- `spikes_in::Accumulator{IT, VT}`: a map of input spike times => current at each time stamp
- `last_spike::IT`: the last time this neuron processed a spike
- `η::F`: post-synaptic (output) spike response function
- `v_th::G`: threshold voltage function
- `last_spike_out::IT`: the last time this neuron released a spike
"""
mutable struct SRM0{VT<:Real, IT<:Integer, F<:Function, G<:Function} <: AbstractNeuron{VT, IT}
    # required fields
    voltage::VT
    spikes_in::Accumulator{IT, VT}
    last_spike::IT

    # model specific fields
    η::F
    v_th::G
    last_spike_out::IT
end

Base.show(io::IO, ::MIME"text/plain", neuron::SRM0) =
    print(io, """SRM0 with $(length(neuron.spikes_in)) queued spikes:
                     voltage: $(neuron.voltage)
                     η:       $(neuron.η)
                     v_th:    $(neuron.v_th)""")
Base.show(io::IO, neuron::SRM0) =
    print(io, "SRM0(v_th: $(neuron.v_th))")

"""
    SRM0(η, v_th)

Create a SRM0 neuron with zero initial voltage and empty spike queue.
"""
SRM0{VT}(η::F, v_th::G) where {VT <: Real, F<:Function, G<:Function} =
    SRM0{VT, Int, F, G}(0, Accumulator{Int, VT}(), 1, η, v_th, 0)

SRM0(η::Function, v_th::VT) where {VT<:Real} = SRM0{VT}(η, Δ -> v_th)

"""
    SRM0(η₀, τᵣ, v_th)

Create a SRM0 neuron with zero initial voltage and empty spike queue by
specifying the response parameters.
"""
function SRM0{VT}(η₀::Real, τᵣ::Real, v_th::Function) where {VT<:Real}
    η = (Δ -> -η₀ * exp(-Δ / τᵣ))
    SRM0{VT}(η, v_th)
end

SRM0(η₀::Real, τᵣ::Real, v_th::VT) where {VT<:Real} = SRM0{VT}(η₀, τᵣ, Δ -> v_th)

"""
    step!(neuron::SRM0, dt::Real = 1.0)::Integer

Evaluate the differential equation between `neuron.last_spike` and the latest input spike.
Return time stamp if the neuron spiked and zero otherwise.
"""
function step!(neuron::SRM0, dt::Real = 1.0)
    # pop the latest spike off the queue
    t = minimum(keys(neuron.spikes_in))
    current_in = DataStructures.reset!(neuron.spikes_in, t)

    # store old voltage
    old_voltage = neuron.voltage

    # evaluate the response function
    neuron.voltage = (neuron.last_spike_out > 0) ? (neuron.η)(dt * (t - neuron.last_spike_out)) : zero(neuron.voltage)

    # accumulate the input spike
    neuron.voltage += current_in

    # choose whether to spike
    spiked = (neuron.voltage >= neuron.v_th(t - neuron.last_spike_out)) && (neuron.voltage - old_voltage > 0)

    # update the last spike
    neuron.last_spike = t + 1
    neuron.last_spike_out = spiked ? t : neuron.last_spike_out

    return spiked ? t : 0
end

"""
    reset!(neuron::SRM0)

Reset the neuron so it never spiked and clear its input spike queue.
"""
function reset!(neuron::SRM0)
    neuron.voltage = 0
    neuron.last_spike = 1
    neuron.last_spike_out = 0
    for key in keys(neuron.spikes_in)
        DataStructures.reset!(neuron.spikes_in, key)
    end
end