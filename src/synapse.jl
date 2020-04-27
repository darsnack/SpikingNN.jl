@reexport module Synapse

using SNNlib.Synapse: delta, alpha, epsp
# using DataStructures: Queue, enqueue!, dequeue!, empty!

import ..SpikingNN: excite!, reset!, isactive

export AbstractSynapse,
       excite!, reset!, isactive

"""
    AbstractSynapse

Inherit from this type to create a concrete synapse.
"""
abstract type AbstractSynapse end

# _ispending(queue, t) = !isempty(queue) && first(queue) <= t
# _ispending(synapse::AbstractSynapse, t) = _ispending(synapse.spikes, t)
# function _shiftspike!(synapse, lastspike, t; dt)
#     while _ispending(synapse, t)
#         lastspike = dequeue!(synapse.spikes) * dt
#     end

#     return lastspike
# end
# function _shiftspike!(synapses::AbstractArray, lastspikes, t; dt)
#     buffer = zeros(eltype(lastspikes), size(lastspikes))
#     pending = map(x -> _ispending(x, t), synapses.spikes)
#     remaining = copy(pending)
#     while any(remaining)
#         @. buffer[remaining] = dequeue!(synapses.spikes[remaining]) * dt
#         remaining = map(x -> _ispending(x, t), synapses.spikes)
#     end

#     lastspikes[pending] .= buffer[pending]

#     return lastspikes
# end

"""
    push!(synapse::AbstractSynapse, spike::Integer)
    push!(synapse::AbstractSynapse, spikes::Vector{<:Integer})

Push a spike(s) into a synapse. The synapse decides how to process this event.
"""
excite!(synapse::Function, spike::Integer) = nothing
excite!(synapse::AbstractSynapse, spikes::Vector{<:Integer}) = map(x -> excite!(synapse, x), spikes)
excite!(synapses::AbstractArray{<:AbstractSynapse}, spikes::Vector{<:Integer}) = map(x -> excite!(synapses, x), spikes)

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

function excite!(synapses::T, spike::Integer) where T<:Union{Delta, AbstractArray{<:Delta}}
    if spike > 0
        synapses.lastspike .= spike
    end
end

# isactive(synapse::Delta, t::Integer; dt::Real = 1.0) = (t * dt == synapse.lastspike) || _ispending(synapse, t)
# isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Delta} =
#     any(t * dt .== synapses.lastspike) || any(map(x -> _ispending(x, t), synapses.spikes))

"""
    (synapse::Delta)(t::Integer; dt::Real = 1.0)

Return `synapse.q` if `t == synapse.lastspike` otherwise return zero.
"""
function (synapse::Delta)(t::Integer; dt::Real = 1.0)
    # synapse.lastspike = _shiftspike!(synapse, synapse.lastspike, t; dt = dt)

    return delta(t * dt, synapse.lastspike, synapse.q)
end
function evalsynapses(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Delta}
    # _shiftspike!(synapses, synapses.lastspike, t; dt = dt)

    return delta(t * dt, synapses.lastspike, synapses.q)
end

function reset!(synapses::T) where T<:Union{Delta, AbstractArray{<:Delta}}
    synapses.lastspike .= -Inf
end

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

function excite!(synapses::T, spike::Integer) where T<:Union{Alpha, AbstractArray{<:Alpha}}
    if spike > 0
        synapses.lastspike .= spike
    end
end

# isactive(synapse::Alpha, t::Real; dt::Real = 1.0) = _ispending(synapse, t) || dt * (t - synapse.lastspike) <= 10 * synapse.τ
# isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Alpha} =
#     any(map(x -> _ispending(x, t), synapses.spikes)) || any(dt .* (t .- synapses.lastspike) .<= 10 .* synapses.τ)

"""
    (synapse::Alpha)(t::Integer; dt::Real = 1.0)

Evaluate an alpha synapse. See [`Synapse.Alpha`](@ref).
"""
function (synapse::Alpha)(t::Integer; dt::Real = 1.0)
    # synapse.lastspike = _shiftspike!(synapse, synapse.lastspike, t; dt = dt)

    return alpha(t * dt, synapse.lastspike, synapse.q, synapse.τ)
end
function evalsynapses(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Alpha}
    # _shiftspike!(synapses, synapses.lastspike, t; dt = dt)

    return alpha(t * dt, synapses.lastspike, synapses.q, synapses.τ)
end

function reset!(synapses::T) where T<:Union{Alpha, AbstractArray{<:Alpha}}
    synapses.lastspike .= -Inf
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
    ϵ₀::VT
    τm::VT
    τs::VT
end
EPSP{IT, VT}(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1) where {IT<:Integer, VT<:Real} = EPSP{IT, VT}(-Inf, ϵ₀, τm, τs)
EPSP(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1) = EPSP{Int, Float32}(ϵ₀ = ϵ₀, τm = τm, τs = τs)

function excite!(synapses::T, spike::Integer) where T<:Union{EPSP, AbstractArray{<:EPSP}}
    if spike > 0
        synapses.lastspike .= spike
    end
end

# isactive(synapse::EPSP, t::Integer; dt::Real) = _ispending(synapse, t) || dt * (t - synapse.lastspike) <= synapse.τs + 8 * synapse.τm
# isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:EPSP} =
#     any(map(x -> _ispending(x, t), synapses.spikes)) || any(dt .* (t .- synapses.lastspike) .<= synapses.τs .+ 8 .* synapses.τm)

"""
    (synapse::EPSP)(t::Integer; dt::Real = 1.0)

Evaluate an EPSP synapse. See [`Synapse.EPSP`](@ref).
"""
function (synapse::EPSP)(t::Integer; dt::Real = 1.0)
    # synapse.lastspike = _shiftspike!(synapse, synapse.lastspike, t; dt = dt)

    return epsp(t * dt, synapse.lastspike, synapse.ϵ₀, synapse.τm, synapse.τs)
end
function evalsynapses(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:EPSP}
    # _shiftspike!(synapses, synapses.lastspike, t; dt = dt)

    return epsp(t * dt, synapses.lastspike, synapses.ϵ₀, synapses.τm, synapses.τs)
end

function reset!(synapses::T) where T<:Union{EPSP, AbstractArray{<:EPSP}}
    synapses.lastspike .= -Inf
end

end