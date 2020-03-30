module Synapse

export excite!

using SNNlib.Synapse: delta, alpha, epsp
using DataStructures
using Base: @kwdef

_ispending(synapse, t) = !isempty(synapse.spikes) && first(synapse.spikes) <= t
function _shiftspike!(synapse, t)
    while _ispending(synapse, t)
        synapse.lastspike = dequeue!(synapse.spikes)
    end

    return synapse
end

"""
    AbstractSynapse

Inherit from this type to create a concrete synapse.
"""
abstract type AbstractSynapse end

"""
    push!(synapse::AbstractSynapse, spike::Integer)
    push!(synapse::AbstractSynapse, spikes::Vector{<:Integer})

Push a spike(s) into a synapse. The synapse decides how to process this event.
"""
excite!(synapse::Function, spike::Integer) = nothing
excite!(synapse::AbstractSynapse, spikes::Vector{<:Integer}) = map(x -> excite!(synapse, x), spikes)

"""
    Delta{IT<:Integer, VT<:Real}

A synapse representing a Dirac-delta at `lastspike`.
"""
@kwdef mutable struct Delta{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::IT = -1
    spikes::Queue{IT} = Queue{Int}()
    q::VT = 1
end

excite!(synapse::Delta, spike::Integer) = enqueue!(synapse.spikes, spike)

isactive(synapse::Delta, t::Integer; dt::Real = 1.0) = (t == synapse.lastspike) || _ispending(synapse, t)

"""
    (synapse::Delta)(t::Integer; dt::Real = 1.0)

Return `synapse.q` if `t == synapse.lastspike` otherwise return zero.
"""
function (synapse::Delta)(t::Integer; dt::Real = 1.0)
    _shiftspike!(synapse, t)

    return delta(t * dt, synapse.lastspike * dt, synapse.q)
end
function evalsynapses(synapses::Vector{T}, t::Integer; dt::Real = 1.0) where T<:Delta
    map(s -> _shiftspike!(s, t), synapses)
    lastspike = map(s -> s.lastspike * dt, synapses)
    q = map(s -> s.q, synapses)


    return delta(t * dt, lastspike, q)
end

"""
    Alpha{IT<:Integer, VT<:Real}

Synapse that returns `(t - lastspike) * (q / τ) * exp(-(t - lastspike - τ) / τ) Θ(t - lastspike)`
(where `Θ` is the Heaviside function).
"""
@kwdef mutable struct Alpha{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::IT = -1
    spikes::Queue{IT} = Queue{Int}()
    q::VT = 1
    τ::VT = 1
end

excite!(synapse::Alpha, spike::Integer) = enqueue!(synapse.spikes, spike)

isactive(synapse::Alpha, t::Real; dt::Real = 1.0) = _ispending(synapse, t) ||
                                                    (synapse.lastspike > 0 && dt * (t - synapse.lastspike) <= 10 * synapse.τ)

"""
    (synapse::Alpha)(t::Integer; dt::Real = 1.0)

Evaluate an alpha synapse. See [`Synapse.Alpha`](@ref).
"""
function (synapse::Alpha)(t::Integer; dt::Real = 1.0)
    _shiftspike!(synapse, t)

    return alpha(t * dt, synapse.lastspike * dt, synapse.q, synapse.τ)
end
function evalsynapses(synapses::Vector{T}, t::Integer; dt::Real = 1.0) where T<:Alpha
    map(s -> _shiftspike!(s, t), synapses)
    lastspike = map(s -> s.lastspike * dt, synapses)
    q = map(s -> s.q, synapses)
    τ = map(s -> s.τ, synapses)

    return alpha(t * dt, lastspike, q, τ)
end

"""
    EPSP{T<:Real}

Synapse that returns `(ϵ₀ / τm - τs) * (exp(-Δ / τm) - exp(-Δ / τs)) Θ(Δ)`
(where `Θ` is the Heaviside function and `Δ = t - lastspike`).

Specifically, this is the EPSP time course for the SRM0 model introduced by Gerstner.
Details: [Spiking Neuron Models: Single Neurons, Populations, Plasticity]
         (https://icwww.epfl.ch/~gerstner/SPNM/node27.html#SECTION02323400000000000000)
"""
@kwdef mutable struct EPSP{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::IT = -1
    spikes::Queue{IT} = Queue{Int}()
    ϵ₀::VT = 1
    τm::VT = 1
    τs::VT = 1
end

excite!(synapse::EPSP, spike::Integer) = enqueue!(synapse.spikes, spike)

isactive(synapse::EPSP, t::Integer; dt::Real) = _ispending(synapse, t) ||
                                                (synapse.lastspike > 0 && dt * (t - synapse.lastspike) <= synapse.τs + 8 * synapse.τm)

"""
    (synapse::EPSP)(t::Integer; dt::Real = 1.0)

Evaluate an EPSP synapse. See [`Synapse.EPSP`](@ref).
"""
function (synapse::EPSP)(t::Integer; dt::Real = 1.0)
    _shiftspike!(synapse, t)

    return epsp(t * dt, synapse.lastspike * dt, synapse.ϵ₀, synapse.τm, synapse.τs)
end
function evalsynapses(synapses::Vector{T}, t::Integer; dt::Real = 1.0) where T<:EPSP
    map(s -> _shiftspike!(s, t), synapses)
    lastspike = map(s -> s.lastspike * dt, synapses)
    ϵ₀ = map(s -> s.ϵ₀, synapses)
    τm = map(s -> s.τm, synapses)
    τs = map(s -> s.τs, synapses)

    return epsp(t * dt, lastspike, ϵ₀, τm, τs)
end

end