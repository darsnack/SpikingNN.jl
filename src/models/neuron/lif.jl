"""
    dlif(v, t, I, vrest, R, tau)
    dlif!(dV::AbstractArray,
          V::AbstractArray,
          t,
          I::AbstractArray,
          vrest::AbstractArray,
          R::AbstractArray,
          tau::AbstractArray)

Evaluate a leaky integrate-and-fire neuron dynamics.

# Fields
- `dv`: the change in membrane potential at time `t`
- `v`: current membrane potential
- `t`: time since last evaluation in seconds
- `I`: external current
- `vrest`: resting membrane potential
- `R`: resistance constant
- `tau`: time constant
"""
dlif(v, I, vrest, R, tau) = vrest + (I * R - v) / tau
function dlif!(dv::AbstractArray,
               v::AbstractArray,
               I::AbstractArray,
               vrest::AbstractArray,
               R::AbstractArray,
               tau::AbstractArray)
    @avx @. dv = vrest + (I * R - v) / tau

    return dv
end
function dlif!(dv::CuArray,
               v::CuArray,
               I::CuArray,
               vrest::CuArray,
               R::CuArray,
               tau::CuArray)
    @. dv = vrest + (I * R - v) / tau

    return dv
end

"""
    LIF(τm, vrest, vreset, R, dims = (1,))
    LIF(; τm, vrest = zero(Float32), vreset = zero(Float32), R = one(Float32), dims = (1,))

A leaky-integrate-fire neuron described by the following differential equation

``v(t) + \\tau_m \\left.\\frac{\\mathrm{d}v}{\\mathrm{d}t}\\right|_t = v_{\\text{rest}} + IR``

When a spike occurs, ``v(t) \\gets v_{\\text{reset}}``

# Fields
- `τm::T`: membrane time constant
- `vrest::T`: the reseting membrane potential
- `vreset::T`: reset voltage potential
- `R::T`: membrane resistive constant (typically = 1)
"""
struct LIF{T<:AbstractArray{<:Real}} <: AbstractCell
    τm::T
    vrest::T
    vreset::T
    R::T
end

LIF(τm, vrest, vreset, R, dims = (1,)) =
    LIF(_fillmemaybe(τm, dims), _fillmemaybe(vrest, dims), _fillmemaybe(vreset, dims), _fillmemaybe(R, dims))
LIF(; τm, vrest = zero(Float32), vreset = zero(Float32), R = one(Float32), dims = (1,)) =
    LIF(τm, vrest, vreset, R, dims)

Base.show(io::IO, ::MIME"text/plain", neuron::LIF{T}) where T =
    print(io, """LIF{$T}:
                     τm:      $(neuron.τm)
                     vrest:   $(neuron.vrest)
                     vreset:  $(neuron.vreset)
                     R:       $(neuron.R)""")
Base.show(io::IO, neuron::LIF{T}) where T =
    print(io, "LIF{$T}(τm = $(neuron.τm), vrest = $(neuron.vrest), vreset = $(neuron.vreset), R = $(neuron.R))")

Base.size(neuron::LIF) = size(neuron.τm)

init(neuron::LIF) = (ComponentArray(voltage = fill!(similar(neuron.τm), 0)),
                     ComponentArray(voltage = fill!(similar(neuron.τm), 0)))

getvoltage(::LIF, state) = state.voltage

function differential!(dstate, state, neuron::LIF, t, currents)
    dlif!(dstate.voltage, state.voltage, currents, neuron.vrest, neuron.R, neuron.τm)

    return dstate
end

function refactor!(state, neuron::LIF, synapses, spikes; dt = 1)
    spiked = spikes .> 0
    state.voltage[spiked] .= neuron.vreset[spiked]
end

"""
    reset!(neuron::LIF)

Reset `neuron` by setting the membrane potential to reset potential.
"""
function reset!(state, neuron::LIF, mask = trues(size(neuron)))
    state.voltage[mask] .= neuron.vreset[mask]
end
