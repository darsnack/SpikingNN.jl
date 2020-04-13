"""
    LIF

A leaky-integrate-fire neuron.

Fields:
- `voltage::VT`: membrane potential
- `current_in::Accumulator{IT, VT}`: a map of time index => current at each time stamp
- `lastspike::IT`: the last time this neuron processed a spike
- `τm::VT`: membrane time constant
- `vreset::VT`: reset voltage potential
- `vth::VT`: threshold voltage potential
- `R::VT`: resistive constant (typically = 1)
"""
mutable struct LIF{VT<:Real, IT<:Integer} <: AbstractCell
    # required fields
    voltage::VT
    current::VT

    # model specific fields
    lastt::IT
    τm::VT
    vreset::VT
    R::VT
end

Base.show(io::IO, ::MIME"text/plain", neuron::LIF) =
    print(io, """LIF:
                     voltage: $(neuron.voltage)
                     current: $(neuron.current)
                     τm:      $(neuron.τm)
                     vreset:  $(neuron.vreset)
                     R:       $(neuron.R)""")
Base.show(io::IO, neuron::LIF) =
    print(io, "LIF(τm: $(neuron.τm), vreset: $(neuron.vreset), R: $(neuron.R))")

"""
    LIF(τm, vreset, vth, R = 1.0)

Create a LIF neuron with zero initial voltage and empty current queue.
"""
LIF(τm::Real, vreset::Real, R::Real = 1.0) = LIF{Float32, Int}(vreset, 0, 0, τm, vreset, R)

"""
    isactive(neuron::LIF, t::Integer)

Return true if the neuron has a current event to process at this time step `t`.
"""
isactive(neuron::LIF, t::Integer; dt::Real = 1.0) = (neuron.current > 0)

getvoltage(neuron::LIF) = neuron.voltage
excite!(neuron::LIF, current) = (neuron.current += current)
excite!(neurons::T, current) where T<:AbstractArray{<:LIF} = (neurons.current .+= current)
spike!(neuron::LIF, t::Integer; dt::Real = 1.0) = (neuron.voltage = neuron.vreset)
spike!(neurons::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:LIF} = (neurons.voltage .= neurons.vreset)

"""
    (neuron::LIF)(t::Integer; dt::Real = 1.0)

Evaluate the neuron model at time `t`.
Return time stamp if the neuron spiked and zero otherwise.
"""
function (neuron::LIF)(t::Integer; dt::Real = 1.0)
    neuron.voltage = SNNlib.Neuron.lif((t - neuron.lastt) * dt, neuron.current, neuron.voltage; R = neuron.R, tau = neuron.τm)
    neuron.lastt = t
    neuron.current = 0

    return neuron.voltage
end
function evalcells(neurons::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:LIF}
    SNNlib.Neuron.lif!((t .- neurons.lastt) .* dt, neurons.current, neurons.voltage; R = neurons.R, tau = neurons.τm)
    neurons.lastt .= t
    neurons.current .= 0

    return neurons.voltage
end

"""
    reset!(neuron::LIF)

Reset the neuron to its reset voltage and clear its input current queue.
"""
function reset!(neuron::LIF)
    neuron.lastt = 0
    neuron.voltage = neuron.vreset
    neuron.current = 0
end
function reset!(neurons::T) where T<:AbstractArray{<:LIF}
    neurons.lastt .= 0
    neurons.voltage .= neurons.vreset
    neurons.current .= 0
end