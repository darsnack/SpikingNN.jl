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
srm0(t, I, eta, tau, tf) = -eta * exp((tf - t) / tau) + I
function srm0!(V::AbstractArray, t, I::AbstractArray, eta::AbstractArray, tau::AbstractArray, tf::AbstractArray)
    @avx @. V = -eta * exp((tf - t) / tau) + I

    return V
end
function srm0!(V::CuArray, t, I::CuArray, eta::CuArray, tau::CuArray, tf::CuArray)
    @. V = -eta * exp((tf - t) / tau) + I

    return V
end

"""
    SRM0(dims::Tuple = (1,); η₀, τr)
    SRM0(dims... = 1; η0, τr)

A SRM0 neuron described by

``v(t) = -\\eta_0 \\exp\\left(-\\frac{t - t^t}{\\tau_r}\\right) \\Theta(t - t^f) + I``

where ``\\Theta`` is the Heaviside function and ``t^f`` is the last output spike time.

*Note:* The SRM0 model normally includes an EPSP term which
  is modeled by [`Synapse.EPSP`](@ref)

For more details see:
  [Spiking Neuron Models: Single Neurons, Populations, Plasticity]
  (https://icwww.epfl.ch/~gerstner/SPNM/node27.html#SECTION02323400000000000000)

Fields:
- `η₀`: refactory potential
- `τr`: refactory time constant
"""
struct SRM0{T<:AbstractArray{<:Real}} <: AbstractCell
    η₀::T
    τr::T
end

SRM0(η₀, τr, dims = (1,)) = SRM0(_fillmemaybe(η₀, dims), _fillmemaybe(τr, dims))
SRM0(; η₀, τr, dims = (1,)) = SRM0(η₀, τr, dims)

Base.show(io::IO, ::MIME"text/plain", neuron::SRM0) =
    print(io, """SRM0:
                     η₀: $(neuron.η₀)
                     τr: $(neuron.τr)""")
Base.show(io::IO, neuron::SRM0) =
    print(io, "SRM0(η₀ = $(neuron.η₀), τr = $(neuron.τr))")

Base.size(neuron::SRM0) = size(neuron.η₀)

init(neuron::SRM0) = (ComponentArray(voltage = fill!(similar(neuron.η₀), 0),
                                     lastspike = fill!(similar(neuron.η₀), -Inf)),
                      nothing)

getvoltage(::SRM0, state) = state.voltage

"""
    evaluate!(dstate, state, neuron::SRM0, t, current; dt = 1)

Evaluate the neuron model at time `t`. Return resulting membrane potential.
"""
function evaluate!(dstate, state, neuron::SRM0, t, current; dt = 1)
    srm0!(state.voltage, t * dt, current, neuron.η₀, neuron.τr, state.lastspike)

    return state
end

function refactor!(state, ::SRM0, synapses, spikes; dt = 1)
    spiked = spikes .> 0
    state.lastspike[spiked] .= spikes[spiked] * dt
    reset!(synapses, repeat(transpose(spiked), size(synapses, 1), 1))
end

"""
    reset!(neuron::SRM0)

Reset `neuron`.
"""
function reset!(state, neuron::SRM0, mask = trues(size(neuron)))
    state.voltage[mask] .= 0
    state.lastspike[mask] .= -Inf
end