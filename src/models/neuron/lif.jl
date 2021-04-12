"""
    lif(V, t, I; vrest, R, tau)
    lif!(V::AbstractArray, t, I::AbstractArray,
         vrest::AbstractArray, R::AbstractArray, tau::AbstractArray)

Evaluate a leaky integrate-and-fire neuron.

# Fields
- `V`: current membrane potential
- `t`: time since last evaluation in seconds
- `I`: external current
- `vrest`: resting membrane potential
- `R`: resistance constant
- `tau`: time constant
"""
lif(V, t, I, vrest, R, tau) = V * exp(-t / tau) + vrest + I * (R / tau)
function lif!(V::AbstractArray, t, I::AbstractArray,
              vrest::AbstractArray, R::AbstractArray, tau::AbstractArray)
    @avx @. V = V * exp(-t / tau) + vrest + I * (R / tau)

    return V
end
function lif!(V::CuArray, t, I::CuArray, vrest::CuArray, R::CuArray, tau::CuArray)
    @. V = V * exp(-t / tau) + vrest + I * (R / tau)

    return V
end

"""
    LIF

A leaky-integrate-fire neuron described by the following differential equation

``\\frac{\\mathrm{d}v}{\\mathrm{d}t} = \\frac{R}{\\tau} I - \\lambda``

# Fields
- `voltage::VT`: membrane potential
- `lastt::IT`: the last time this neuron processed a spike
- `τm::VT`: membrane time constant
- `vreset::VT`: reset voltage potential
- `R::VT`: resistive constant (typically = 1)
"""
mutable struct LIF{VT<:Real, IT<:Integer} <: AbstractCell
    # model state
    voltage::VT
    lastt::IT

    # model parameters
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
    print(io, "LIF(voltage = $(neuron.voltage), current = $(neuron.current))")

"""
    LIF(τm, vreset, R = 1.0)

Create a LIF neuron with zero initial voltage and empty current queue.
"""
LIF{VT, IT}(;τm::Real, vreset::Real = zero(VT), R::Real = zero(VT)) where {VT<:Real, IT<:Integer} =
    LIF{VT, IT}(vreset, 0, τm, vreset, R)
LIF(;kwargs...) = LIF{Float32, Int}(;kwargs...)

isactive(neuron::LIF, t::Integer; dt::Real = 1.0) = true
getvoltage(neuron::LIF) = neuron.voltage

"""
    spike!(neuron::LIF, t::Integer; dt::Real = 1.0)
    spike!(neurons::AbstractArray{<:LIF}, spikes; dt::Real = 1.0)

Record a output spike from the threshold function with the `neuron` body.
Sets `neuron.lastt`.
"""
spike!(neuron::LIF, t::Integer; dt::Real = 1.0) = (t > 0) && (neuron.voltage = neuron.vreset)
function spike!(neurons::T, spikes; dt::Real = 1.0) where T<:AbstractArray{<:LIF}
    map!((x, y, s) -> (s > 0) ? y : x, neurons.voltage, neurons.voltage, neurons.vreset, spikes)
end



"""
    evaluate!(neuron::LIF, t::Integer; dt::Real = 1.0)
    (neuron::LIF)(t::Integer; dt::Real = 1.0)
    evaluate!(neurons::AbstractArray{<:LIF}, t::Integer; dt::Real = 1.0)

Evaluate the neuron model at time `t`. Return the resulting membrane potential.
"""
function evaluate!(neuron::LIF, t::Integer, current; dt::Real = 1.0)
    neuron.voltage =
        lif(neuron.voltage, (t - neuron.lastt) * dt, current, 0, neuron.R, neuron.τm)
    neuron.lastt = t

    return neuron.voltage
end
(neuron::LIF)(t::Integer, current; dt::Real = 1.0) = evaluate!(neuron, t, current; dt = dt)
function evaluate!(neurons::T, t::Integer, current; dt::Real = 1.0) where T<:AbstractArray{<:LIF}
    lif!(neurons.voltage,
         (t .- neurons.lastt) .* dt,
         current,
         zero(neurons.voltage),
         neurons.R,
         neurons.τm)
    neurons.lastt .= t

    return neurons.voltage
end

function refactor!(neuron::LIF, synapses, t; dt = 1.0)
    neuron.voltage = neuron.vreset
    
    return neuron
end
function refactor!(neurons::AbstractVector{<:LIF}, synapses, spikes; dt = 1.0)
    spiked = spikes .> 0
    neurons.voltage[spiked] .= neurons.vreset[spiked]
    
    return neurons
end

"""
    reset!(neuron::LIF)
    reset!(neurons::AbstractArray{<:LIF})

Reset `neuron` by setting the membrane potential to `neuron.vreset`.
"""
function reset!(neuron::LIF)
    neuron.lastt = 0
    neuron.voltage = neuron.vreset
end
function reset!(neurons::T) where T<:AbstractArray{<:LIF}
    neurons.lastt .= 0
    neurons.voltage .= neurons.vreset
end