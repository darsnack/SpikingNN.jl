"""
    LIF

A leaky-integrate-fire neuron described by the following differential equation

``\\tau \\frac{\\mathrm{d}v}{\\mathrm{d}t} = RI - (v - v_{\\text{rest}})``

After spiking, the membrane potential is clamped to `vreset` for `τr` seconds.

# Fields
- `voltage::VT`: membrane potential
- `current::VT`: injected (unprocessed) current
- `lastt::IT`: the last time this neuron was processed
- `lastspike::IT`: the last time this neuron spiked
- `τm::VT`: membrane time constant
- `τr::VT`: refractory time constant
- `vrest::VT`: resting membrane potential
- `vreset::VT`: reset membrane potential
- `R::VT`: resistive constant (typically = 1)
"""
mutable struct LIF{VT<:Real, IT<:Integer} <: AbstractCell
    voltage::VT
    current::VT
    lastt::IT
    lastspike::VT
    τm::VT
    τr::VT
    vrest::VT
    vreset::VT
    R::VT
end

Base.show(io::IO, ::MIME"text/plain", neuron::LIF) =
    print(io, """LIF:
                     voltage: $(neuron.voltage)
                     current: $(neuron.current)
                     τm:      $(neuron.τm)
                     τr:      $(neuron.τr)
                     vrest:   $(neuron.vrest)
                     vreset:  $(neuron.vreset)
                     R:       $(neuron.R)
                 """)
Base.show(io::IO, neuron::LIF) =
    print(io, "LIF(voltage = $(neuron.voltage), current = $(neuron.current))")

"""
    LIF(;τm, τr = 0.0, vrest = 0.0, vreset = 0.0, R = 1.0)

Create a LIF neuron.
"""
LIF(;τm::Real, τr::Real = 0.0, vrest::Real = 0.0, vreset::Real = 0.0, R::Real = 1.0) =
    LIF{Float32, Int}(vreset, 0, 0, -Inf, τm, τr, vrest, vreset, R)

isactive(neuron::LIF, t::Integer; dt::Real = 1.0) = (neuron.current > 0)
getvoltage(neuron::LIF) = neuron.voltage

"""
    excite!(neuron::LIF, current)
    excite!(neurons::AbstractArray{<:LIF}, current)

Excite a `neuron` with external `current`.
"""
excite!(neuron::LIF, current) = (neuron.current += current)
excite!(neurons::T, current) where T<:AbstractArray{<:LIF} = (neurons.current .+= current)

"""
    spike!(neuron::LIF, t::Integer; dt::Real = 1.0)
    spike!(neurons::AbstractArray{<:LIF}, spikes; dt::Real = 1.0)

Record a output spike from the threshold function with the `neuron` body.
Sets `neuron.lastt`.
"""
function spike!(neuron::LIF, t::Integer; dt::Real = 1.0)
    if t > 0
        neuron.voltage = neuron.vreset
        neuron.lastspike = t * dt
    end
end
function spike!(neurons::T, spikes; dt::Real = 1.0) where T<:AbstractArray{<:LIF}
    map!((x, y, s) -> (s > 0) ? y : x, neurons.voltage, neurons.voltage, neurons.vreset, spikes)
    map!((tf, s) -> (s > 0) ? s * dt : tf, neurons.lastspike, neurons.lastspike, spikes)
end

"""
    evaluate!(neuron::LIF, t::Integer; dt::Real = 1.0)
    (neuron::LIF)(t::Integer; dt::Real = 1.0)
    evaluate!(neurons::AbstractArray{<:LIF}, t::Integer; dt::Real = 1.0)

Evaluate the neuron model at time `t`. Return the resulting membrane potential.
"""
function evaluate!(neuron::LIF, t::Integer; dt::Real = 1.0)
    # clamp
    if t * dt - neuron.lastspike < neuron.τr
        neuron.lastt = t
        neuron.current = 0

        return neuron.voltage
    end

    neuron.voltage = SpikingNNFunctions.Neuron.lif((t - neuron.lastt) * dt, neuron.current, neuron.voltage;
                                                   vrest = neuron.vrest, R = neuron.R, tau = neuron.τm)
    neuron.lastt = t
    neuron.current = 0

    return neuron.voltage
end
(neuron::LIF)(t::Integer; dt::Real = 1.0) = evaluate!(neuron, t; dt = dt)
function evaluate!(neurons::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:LIF}
    SpikingNNFunctions.Neuron.lif!((t .- neurons.lastt) .* dt, neurons.current, neurons.voltage;
                                   vrest = neurons.vrest, R = neurons.R, tau = neurons.τm)
    map!((v, tf, τr) -> (t * dt - tf) < τr ? zero(v) : v,
         neurons.voltage, neurons.voltage, neurons.lastspike, neurons.τr)
    neurons.lastt .= t
    neurons.current .= 0

    return neurons.voltage
end

"""
    reset!(neuron::LIF)
    reset!(neurons::AbstractArray{<:LIF})

Reset `neuron` by setting the membrane potential to `neuron.vreset`.
"""
function reset!(neuron::LIF)
    neuron.lastt = 0
    neuron.lastspike = -Inf
    neuron.voltage = neuron.vreset
    neuron.current = 0
end
function reset!(neurons::T) where T<:AbstractArray{<:LIF}
    neurons.lastt .= 0
    neurons.lastspike .= -Inf
    neurons.voltage .= neurons.vreset
    neurons.current .= 0
end