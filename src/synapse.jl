@reexport module Synapse

using SNNlib.Synapse: delta, alpha, epsp
using DataStructures: Queue, enqueue!, dequeue!, empty!

import ..SpikingNN: excite!, reset!, isactive

export AbstractSynapse, QueuedSynapse,
       excite!, reset!, isactive

"""
    AbstractSynapse

Inherit from this type to create a concrete synapse.
"""
abstract type AbstractSynapse end

_ispending(queue, t) = !isempty(queue) && first(queue) <= t
function _shiftspike!(queue, lastspike, t)
    while _ispending(queue, t)
        lastspike = dequeue!(queue)
    end

    return lastspike
end
function _shiftspike!(queues::AbstractArray, lastspikes, t)
    pending = map(x -> _ispending(x, t), queues)
    while any(pending)
        @. lastspikes[pending] = dequeue!(queues[pending])
        pending = map(x -> _ispending(x, t), queues)
    end

    return lastspikes
end

"""
    push!(synapse::AbstractSynapse, spike::Integer)
    push!(synapse::AbstractSynapse, spikes::Vector{<:Integer})

Push a spike(s) into a synapse. The synapse decides how to process this event.
"""
excite!(synapse::Function, spike::Integer) = nothing
excite!(synapse::AbstractSynapse, spikes::Vector{<:Integer}) = map(x -> excite!(synapse, x), spikes)
excite!(synapses::AbstractArray{<:AbstractSynapse}, spikes::Vector{<:Integer}) = map(x -> excite!(synapses, x), spikes)

struct QueuedSynapse{ST<:AbstractSynapse, IT<:Integer} <: AbstractSynapse
    core::ST
    queue::Queue{IT}
end
QueuedSynapse{IT}(synapse) where {IT<:Integer} = QueuedSynapse{typeof(synapse), IT}(synapse, Queue{IT}())
QueuedSynapse(synapse) = QueuedSynapse{typeof(synapse), Int}(synapse, Queue{Int}())

excite!(synapse::QueuedSynapse, spike::Integer) = enqueue!(synapse.queue, spike)
excite!(synapses::T, spike::Integer) where T<:AbstractArray{<:QueuedSynapse} =
    map(x -> enqueue!(x, spike), synapses.queue)

isactive(synapse::QueuedSynapse, t::Integer; dt::Real = 1.0) = _ispending(synapse.queue, t) || isactive(synapse.core, t; dt = dt)
isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:QueuedSynapse} =
    any(map(x -> _ispending(x, t), synapses.queue)) || isactive(synapses.core, t; dt = dt)

function (synapse::QueuedSynapse)(t::Integer; dt::Real = 1.0)
    excite!(synapse.core, _shiftspike!(synapse.queue, 0, t))

    return synapse.core(t; dt = dt)
end
function evalsynapses(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:QueuedSynapse}
    lastspikes = _shiftspike!(synapses.queue, zeros(Int, size(synapses)), t)
    @inbounds for i in eachindex(synapses)
        excite!(view(synapses.core, i), lastspikes[i])
    end

    return evalsynapses(synapses.core, t; dt = dt)
end

function reset!(synapse::QueuedSynapse)
    empty!(synapse.queue)
    reset!(synapse.core)
end
function reset!(synapses::T) where T<:AbstractArray{<:QueuedSynapse}
    empty!.(synapses.queue)
    reset!(synapses.core)
end

"""
    Delta{IT<:Integer, VT<:Real}

A synapse representing a Dirac-delta at `lastspike`.
"""
mutable struct Delta{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::VT
    q::VT
end
Delta{IT, VT}(;q::Real = 1) where {IT<:Integer, VT<:Real} = Delta{IT, VT}(-Inf, q)
Delta(;q::Real = 1) = Delta{Int, Float32}(q = q)

excite!(synapse::Delta, spike::Integer) = (spike > 0) && (synapse.lastspike = spike)
excite!(synapses::T, spike::Integer) where T<:AbstractArray{<:Delta} = (spike > 0) && (synapses.lastspike .= spike)

isactive(synapse::Delta, t::Integer; dt::Real = 1.0) = (t * dt == synapse.lastspike)
isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Delta} = any(t * dt .== synapses.lastspike)

"""
    (synapse::Delta)(t::Integer; dt::Real = 1.0)

Return `synapse.q` if `t == synapse.lastspike` otherwise return zero.
"""
function (synapse::Delta)(t::Integer; dt::Real = 1.0)
    # synapse.lastspike = _shiftspike!(synapse, synapse.lastspike, t; dt = dt)

    return delta(t * dt, synapse.lastspike * dt, synapse.q)
end
function evalsynapses(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Delta}
    # _shiftspike!(synapses, synapses.lastspike, t; dt = dt)

    return delta(t * dt, synapses.lastspike * dt, synapses.q)
end

reset!(synapse::Delta) = (synapse.lastspike = -Inf)
reset!(synapses::T) where T<:AbstractArray{<:Delta}= (synapses.lastspike .= -Inf)

"""
    Alpha{IT<:Integer, VT<:Real}

Synapse that returns `(t - lastspike) * (q / τ) * exp(-(t - lastspike - τ) / τ) Θ(t - lastspike)`
(where `Θ` is the Heaviside function).
"""
mutable struct Alpha{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::VT
    q::VT
    τ::VT
end
Alpha{IT, VT}(;q::Real = 1, τ::Real = 1) where {IT<:Integer, VT<:Real} = Alpha{IT, VT}(-Inf, q, τ)
Alpha(;q::Real = 1, τ::Real = 1) = Alpha{Int, Float32}(q = q, τ = τ)

excite!(synapse::Alpha, spike::Integer) = (spike > 0) && (synapse.lastspike = spike)
excite!(synapses::T, spike::Integer) where T<:AbstractArray{<:Alpha} = (spike > 0) && (synapses.lastspike .= spike)

isactive(synapse::Alpha, t::Real; dt::Real = 1.0) = dt * (t - synapse.lastspike) <= 10 * synapse.τ
isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Alpha} =
    any(dt .* (t .- synapses.lastspike) .<= 10 .* synapses.τ)

"""
    (synapse::Alpha)(t::Integer; dt::Real = 1.0)

Evaluate an alpha synapse. See [`Synapse.Alpha`](@ref).
"""
function (synapse::Alpha)(t::Integer; dt::Real = 1.0)
    # synapse.lastspike = _shiftspike!(synapse, synapse.lastspike, t; dt = dt)

    return alpha(t * dt, synapse.lastspike * dt, synapse.q, synapse.τ)
end
function evalsynapses(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Alpha}
    # _shiftspike!(synapses, synapses.lastspike, t; dt = dt)

    return alpha(t * dt, synapses.lastspike * dt, synapses.q, synapses.τ)
end

reset!(synapse::Alpha) = (synapse.lastspike = -Inf)
reset!(synapses::T) where T<:AbstractArray{<:Alpha}= (synapses.lastspike .= -Inf)

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
    ϵ₀::VT
    τm::VT
    τs::VT
end
EPSP{IT, VT}(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1) where {IT<:Integer, VT<:Real} = EPSP{IT, VT}(-Inf, ϵ₀, τm, τs)
EPSP(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1) = EPSP{Int, Float32}(ϵ₀ = ϵ₀, τm = τm, τs = τs)

excite!(synapse::EPSP, spike::Integer) = (spike > 0) && (synapse.lastspike = spike)
excite!(synapses::T, spike::Integer) where T<:AbstractArray{<:EPSP} = (spike > 0) && (synapses.lastspike .= spike)

isactive(synapse::EPSP, t::Integer; dt::Real) = dt * (t - synapse.lastspike) <= synapse.τs + 8 * synapse.τm
isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:EPSP} =
    any(dt .* (t .- synapses.lastspike) .<= synapses.τs .+ 8 .* synapses.τm)

"""
    (synapse::EPSP)(t::Integer; dt::Real = 1.0)

Evaluate an EPSP synapse. See [`Synapse.EPSP`](@ref).
"""
function (synapse::EPSP)(t::Integer; dt::Real = 1.0)
    # synapse.lastspike = _shiftspike!(synapse, synapse.lastspike, t; dt = dt)

    return epsp(t * dt, synapse.lastspike * dt, synapse.ϵ₀, synapse.τm, synapse.τs)
end
function evalsynapses(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:EPSP}
    # _shiftspike!(synapses, synapses.lastspike, t; dt = dt)

    return epsp(t * dt, synapses.lastspike * dt, synapses.ϵ₀, synapses.τm, synapses.τs)
end

reset!(synapse::EPSP) = (synapse.lastspike = -Inf)
reset!(synapses::T) where T<:AbstractArray{<:EPSP}= (synapses.lastspike .= -Inf)

end