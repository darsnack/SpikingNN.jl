@reexport module Synapse

using SNNlib.Synapse: delta, alpha, epsp
using DataStructures: Queue, enqueue!, dequeue!, empty!

import ..SpikingNN: excite!, reset!, isactive

export AbstractSynapse,
       excite!, reset!, isactive

_ispending(synapse, t) = !isempty(synapse.spikes) && first(synapse.spikes) <= t
function _shiftspike!(synapse, lastspike, t; dt)
    while _ispending(synapse, t)
        lastspike = dequeue!(synapse.spikes) * dt
    end

    return lastspike
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
mutable struct Delta{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::VT
    spikes::Queue{IT}
    q::VT
end
Delta{IT, VT}(;q::Real = 1) where {IT<:Integer, VT<:Real} = Delta{IT, VT}(-Inf, Queue{IT}(), q)
Delta(;q::Real = 1) = Delta{Int, Float32}(q = q)

excite!(synapse::Delta, spike::Integer) = enqueue!(synapse.spikes, spike)

isactive(synapse::Delta, t::Integer; dt::Real = 1.0) = (t * dt == synapse.lastspike) || _ispending(synapse, t)

"""
    (synapse::Delta)(t::Integer; dt::Real = 1.0)

Return `synapse.q` if `t == synapse.lastspike` otherwise return zero.
"""
function (synapse::Delta)(t::Integer; dt::Real = 1.0)
    synapse.lastspike = _shiftspike!(synapse, synapse.lastspike, t; dt = dt)

    return delta(t * dt, synapse.lastspike, synapse.q)
end
function evalsynapses(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Delta}
    @inbounds for i in eachindex(synapses)
        synapses.lastspike[i] = _shiftspike!(synapses[i], synapses.lastspike[i], t; dt = dt)
    end

    return delta(t * dt, synapses.lastspike, synapses.q)
end

function reset!(synapse::Delta)
    synapse.lastspike = -Inf
    empty!(synapse.spikes)
end
function reset!(synapses::T) where T<:AbstractArray{<:Delta}
    synapses.lastspike .= -Inf
    empty!.(synapses.spikes)
end

"""
    Alpha{IT<:Integer, VT<:Real}

Synapse that returns `(t - lastspike) * (q / τ) * exp(-(t - lastspike - τ) / τ) Θ(t - lastspike)`
(where `Θ` is the Heaviside function).
"""
mutable struct Alpha{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::VT
    spikes::Queue{IT}
    q::VT
    τ::VT
end
Alpha{IT, VT}(;q::Real = 1, τ::Real = 1) where {IT<:Integer, VT<:Real} = Alpha{IT, VT}(-Inf, Queue{IT}(), q, τ)
Alpha(;q::Real = 1, τ::Real = 1) = Alpha{Int, Float32}(q = q, τ = τ)

excite!(synapse::Alpha, spike::Integer) = enqueue!(synapse.spikes, spike)

isactive(synapse::Alpha, t::Real; dt::Real = 1.0) = _ispending(synapse, t) || dt * (t - synapse.lastspike) <= 10 * synapse.τ

"""
    (synapse::Alpha)(t::Integer; dt::Real = 1.0)

Evaluate an alpha synapse. See [`Synapse.Alpha`](@ref).
"""
function (synapse::Alpha)(t::Integer; dt::Real = 1.0)
    synapse.lastspike = _shiftspike!(synapse, synapse.lastspike, t; dt = dt)

    return alpha(t * dt, synapse.lastspike, synapse.q, synapse.τ)
end
function evalsynapses(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Alpha}
    @inbounds for i in eachindex(synapses)
        synapses.lastspike[i] = _shiftspike!(synapses[i], synapses.lastspike[i], t; dt = dt)
    end

    return alpha(t * dt, synapses.lastspike, synapses.q, synapses.τ)
end

function reset!(synapse::Alpha)
    synapse.lastspike = -Inf
    empty!(synapse.spikes)
end
function reset!(synapses::T) where T<:AbstractArray{<:Alpha}
    synapses.lastspike .= -Inf
    empty!.(synapses.spikes)
end

"""
    EPSP{T<:Real}

Synapse that returns `(ϵ₀ / τm - τs) * (exp(-Δ / τm) - exp(-Δ / τs)) Θ(Δ)`
(where `Θ` is the Heaviside function and `Δ = t - lastspike`).

Specifically, this is the EPSP time course for the SRM0 model introduced by Gerstner.
Details: [Spiking Neuron Models: Single Neurons, Populations, Plasticity]
         (https://icwww.epfl.ch/~gerstner/SPNM/node27.html#SECTION02323400000000000000)
"""
mutable struct EPSP{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::VT
    spikes::Queue{IT}
    ϵ₀::VT
    τm::VT
    τs::VT
end
EPSP{IT, VT}(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1) where {IT<:Integer, VT<:Real} = EPSP{IT, VT}(-Inf, Queue{IT}(), ϵ₀, τm, τs)
EPSP(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1) = EPSP{Int, Float32}(ϵ₀ = ϵ₀, τm = τm, τs = τs)

excite!(synapse::EPSP, spike::Integer) = enqueue!(synapse.spikes, spike)

isactive(synapse::EPSP, t::Integer; dt::Real) = _ispending(synapse, t) || dt * (t - synapse.lastspike) <= synapse.τs + 8 * synapse.τm

"""
    (synapse::EPSP)(t::Integer; dt::Real = 1.0)

Evaluate an EPSP synapse. See [`Synapse.EPSP`](@ref).
"""
function (synapse::EPSP)(t::Integer; dt::Real = 1.0)
    synapse.lastspike = _shiftspike!(synapse, synapse.lastspike, t; dt = dt)

    return epsp(t * dt, synapse.lastspike, synapse.ϵ₀, synapse.τm, synapse.τs)
end
function evalsynapses(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:EPSP}
    @inbounds for i in eachindex(synapses)
        synapses.lastspike[i] = _shiftspike!(synapses[i], synapses.lastspike[i], t; dt = dt)
    end

    return epsp(t * dt, synapses.lastspike, synapses.ϵ₀, synapses.τm, synapses.τs)
end

function reset!(synapse::EPSP)
    synapse.lastspike = -Inf
    empty!(synapse.spikes)
end
function reset!(synapses::T) where T<:AbstractArray{<:EPSP}
    synapses.lastspike .= -Inf
    empty!.(synapses.spikes)
end

end