"""
    srm0(V, t, I, lastspike, eta)
    srm0!(V::AbstractArray, t, I::AbstractArray, lastspike::AbstractArray, eta)

Evaluate a SRM0 neuron.

# Fields
- `t`: current time in seconds
- `I`: external current
- `V`: current membrane potential
- `lastspike`: time of last output spike in seconds
- `eta`: post-synaptic response function
"""
srm0(V, t, I, lastspike, eta) = eta(t - lastspike) + I
function srm0!(V::AbstractArray, t, I::AbstractArray, lastspike::AbstractArray, eta)
    @. V = map(eta, (t - lastspike)) + I
    
    return V
end

"""
    SRM0

A SRM0 neuron described by

``v(t) = \\eta(t - t^f) \\Theta(t - t^f) + I``

  where ``\\Theta`` is the Heaviside function and
  ``t^f`` is the last output spike time.

*Note:* The SRM0 model normally includes an EPSP term which
  is modeled by [`Synapse.EPSP`](@ref)

For more details see:
  [Spiking Neuron Models: Single Neurons, Populations, Plasticity]
  (https://icwww.epfl.ch/~gerstner/SPNM/node27.html#SECTION02323400000000000000)

Fields:
- `voltage::VT`: membrane potential
- `current::VT`: injected (unprocessed) current
- `lastspike::VT`: last time this neuron spiked
- `η::F`: refractory response function
"""
mutable struct SRM0{VT<:Real, F<:Function} <: AbstractCell
    # model state
    voltage::VT
    current::VT
    lastspike::VT
    
    # model parameters
    η::F
end

Base.show(io::IO, ::MIME"text/plain", neuron::SRM0) =
    print(io, """SRM0:
                     voltage:   $(neuron.voltage)
                     current:   $(neuron.current)
                     lastspike: $(neuron.lastspike)
                     η:         $(neuron.η)""")
Base.show(io::IO, neuron::SRM0) =
    print(io, "SRM0(voltage: $(neuron.voltage), current = $(neuron.current))")

"""
    SRM0{T}(η) where T<:Real
    SRM0(η)

Create a SRM0 neuron with refractory response function `η`.
"""
SRM0{T}(η::F) where {T<:Real, F<:Function} = SRM0{T, F}(0, 0, -Inf, η)
SRM0(args...) = SRM0{Float32}(args...)

"""
    SRM0(η₀, τᵣ, v_th)

Create a SRM0 neuron with refractory response function:

``-\\eta_0 \\exp\\left(-\\frac{\\Delta}{\\tau_r}\\right)``
"""
function SRM0{T}(;η₀::Real, τᵣ::Real) where T<:Real
    η = Δ -> @avx -η₀ * exp(-Δ / τᵣ)
    SRM0{T}(η)
end
SRM0(;kwargs...) = SRM0{Float32}(;kwargs...)

isactive(neuron::SRM0, t::Integer; dt::Real = 1.0) = (neuron.current > 0)
getvoltage(neuron::SRM0) = neuron.voltage

"""
    excite!(neuron::SRM0, current)
    excite!(neurons::AbstractArray{<:SRM0}, current)

Excite an SRM0 `neuron` with external `current`.
"""
excite!(neuron::SRM0, current) = (neuron.current += current)
excite!(neurons::T, current) where T<:AbstractArray{<:SRM0} = (neurons.current .+= current)

"""
    spike!(neuron::SRM0, t::Integer; dt::Real = 1.0)
    spike!(neurons::AbstractArray{<:SRM0}, spikes; dt::Real = 1.0)

Record a output spike from the threshold function with the `neuron` body.
Sets `neuron.lastspike`.
"""
spike!(neuron::SRM0, t::Integer; dt::Real = 1.0) = (t > 0) && (neuron.lastspike = dt * t)
function spike!(neurons::T, spikes; dt::Real = 1.0) where T<:AbstractArray{<:SRM0}
    map!((x, s) -> (s > 0) ? dt * s : x, neurons.lastspike, neurons.lastspike, spikes)
end

"""
    evaluate!(neuron::SRM0, t::Integer; dt::Real = 1.0)
    (neuron::SRM0)(t::Integer; dt::Real = 1.0)
    evaluate!(neurons::AbstractArray{<:SRM0}, t::Integer; dt::Real = 1.0)

Evaluate the neuron model at time `t`. Return resulting membrane potential.
"""
function evaluate!(neuron::SRM0, t::Integer; dt::Real = 1.0)
    neuron.voltage = srm0(neuron.voltage, t * dt, neuron.current, neuron.lastspike, neuron.η)
    neuron.current = 0

    return neuron.voltage
end
(neuron::SRM0)(t::Integer; dt::Real = 1.0) = evaluate!(neuron, t; dt = dt)
function evaluate!(neurons::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:SRM0}
    srm0!(neurons.voltage, t * dt, neurons.current, neurons.lastspike, neurons.η)
    neurons.current .= 0

    return neurons.voltage
end

"""
    reset!(neuron::SRM0)
    reset!(neurons::AbstractArray{<:SRM0})

Reset `neuron`.
"""
function reset!(neuron::SRM0)
    neuron.voltage = 0
    neuron.lastspike = -Inf
    neuron.current = 0
end
function reset!(neurons::T) where T<:AbstractArray{<:SRM0}
    neurons.voltage .= 0
    neurons.lastspike .= -Inf
    neurons.current .= 0
end