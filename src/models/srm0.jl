using .Threshold: isactive

"""
    SRM0

A SRM0 neuron.

Fields:
- `voltage::VT`: membrane potential
- `current_in::Accumulator{IT, VT}`: a map of time index => current at each time stamp
- `η::F`: post-synaptic (output) spike response function
- `v_th::G`: threshold voltage function
- `last_spike_out::IT`: the last time this neuron released a spike
"""
mutable struct SRM0{VT<:Real, IT<:Integer, F<:Function, G} <: AbstractNeuron{VT, IT}
    # required fields
    voltage::VT
    current_in::Accumulator{IT, VT}

    # model specific fields
    η::F
    threshfunc::G
    last_spike_out::IT
end

Base.show(io::IO, ::MIME"text/plain", neuron::SRM0) =
    print(io, """SRM0 with $(length(neuron.current_in)) queued current events:
                     voltage:    $(neuron.voltage)
                     η:          $(neuron.η)
                     threshfunc: $(neuron.threshfunc)""")
Base.show(io::IO, neuron::SRM0) =
    print(io, "SRM0(voltage: $(neuron.voltage))")

"""
    SRM0(η, v_th)

Create a SRM0 neuron with zero initial voltage and empty current queue.
"""
SRM0{VT}(η::F, v_th::G) where {VT<:Real, F<:Function, G} =
    SRM0{VT, Int, F, G}(0, Accumulator{Int, VT}(), η, v_th, 0)

SRM0(η::Function, v_th::VT) where {VT<:Real} = SRM0{VT}(η, (Δ, v) -> v >= v_th)

"""
    SRM0(η₀, τᵣ, v_th)

Create a SRM0 neuron with zero initial voltage and empty current queue by
specifying the response parameters.
"""
function SRM0{VT}(η₀::Real, τᵣ::Real, v_th) where {VT<:Real}
    η = (Δ -> -η₀ * exp(-Δ / τᵣ))
    SRM0{VT}(η, v_th)
end

SRM0(η₀::Real, τᵣ::Real, v_th::VT) where {VT<:Real} = SRM0{VT}(η₀, τᵣ, (Δ, v) -> v >= v_th)

"""
    isactive(neuron::SRM0, t::Integer)

Return true if the neuron has a current event to process at this time step `t` or threshold
function is active.
"""
isactive(neuron::SRM0, t::Integer) = true

"""
    (neuron::SRM0)(t::Integer; dt::Real = 1.0)

Evaluate the neuron model at time `t`.
Return time stamp if the neuron spiked and zero otherwise.
"""
function (neuron::SRM0)(t::Integer; dt::Real = 1.0)
    # pop the latest spike off the queue
    current_in = DataStructures.reset!(neuron.current_in, t)

    # store old voltage
    old_voltage = neuron.voltage

    # evaluate the response function
    neuron.voltage = (neuron.last_spike_out > 0) ? (neuron.η)(dt * (t - neuron.last_spike_out)) : zero(neuron.voltage)

    # accumulate the input spike
    neuron.voltage += current_in

    # choose whether to spike
    spiked = neuron.threshfunc(dt * (t - neuron.last_spike_out), neuron.voltage) && (neuron.voltage - old_voltage > 0)

    # update the last spike
    neuron.last_spike_out = spiked ? t : neuron.last_spike_out

    # clear current cue
    spiked && map(x -> DataStructures.reset!(neuron.current_in, x), collect(keys(neuron.current_in)))

    return spiked ? t : 0
end

"""
    reset!(neuron::SRM0)

Reset the neuron so it never spiked and clear its input spike queue.
"""
function reset!(neuron::SRM0)
    neuron.voltage = 0
    neuron.last_spike_out = 0
    for key in keys(neuron.current_in)
        DataStructures.reset!(neuron.current_in, key)
    end
end