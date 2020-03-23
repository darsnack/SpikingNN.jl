module Synapse

using UnPack
using Base: @kwdef

# """
#     SynapseParameters

# A data structure to store the timing parameters of a synapse.

# Fields:
# - `Tw::Real`: the length of time that the response function is non-zero
#     (i.e. the response will be sampled over the interval `[0, T_window]`)
# - `dt::Real`: the sampling rate
# - `samples::Vector{Real}`: an array to store samples (to be populated by `sampleresponse!()`)
# """
# struct SynapseParameters{T<:Real}
#     Tw::T
#     dt::T
#     samples::Vector{T}

#     function SynapseParameters{T}(Tw, dt, samples::Vector{T}) where {T<:Real}
#         (cld(Tw, dt) != length(samples)) && error("SynapseParameters requires ceil(Tw, dt) == length(samples) (received Tw = $Tw, dt = $dt, length = $(length(samples))")

#         new{T}(Tw, dt, samples)
#     end
# end
# SynapseParameters{T}(Tw::Real, dt::Real) where {T<:Real} = SynapseParameters{T}(Tw, dt, zeros(T, Int(cld(Tw, dt))))

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
push!(synapse::Function, spike::Integer) = nothing
push!(synapse::AbstractSynapse, spikes::Vector{<:Integer}) = map(x -> push!(synapse, x), spikes)

# """
#     sampleresponse(response)
#     sampleresponse!(response)

# Return the vector of samples representing the response function.
# Call `sampleresponse!()` to resample the function.
# """
# sampleresponse(response::AbstractSynapse) = response.params.samples, length(response.params.samples)
# function sampleresponse!(response::AbstractSynapse)
#     @unpack Tw, dt = response.params
#     N = length(response.params.samples)
#     t = collect(1:N)
#     response.params.samples .= response.(dt .* t .- dt)

#     return response.params.samples, N
# end

"""
    Delta{IT<:Integer, VT<:Real}

A synapse representing a Dirac-delta at `lastspike`.
"""
@kwdef struct Delta{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::IT = 0
    q::VT = 1
end
Delta(lastspike = 0, q = 1) = Delta{Int, Float32}(lastspike = lastspike, q = q)

push!(synapse::Delta, spike::Integer) = (synapse.lastspike = spike)

"""
    (synapse::Delta)(t::Real; dt::Real = 1.0)

Return `synapse.q` if `t == synapse.lastspike` otherwise return zero.
"""
(synapse::Delta)(t::Real; dt::Real = 1.0) = (t == synapse.lastspike) ? synapse.q : zero(synapse.q)

"""
    Alpha{IT<:Integer, VT<:Real}

Synapse that returns `(t - lastspike) * (q / τ) * exp(-(t - lastspike - τ) / τ) Θ(t - lastspike)`
(where `Θ` is the Heaviside function).
"""
@kwdef struct Alpha{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::IT = 0
    q::VT = 1
    τ::VT = 1
end
Alpha(lastspike = 0, q = 1, τ = 1) = Alpha{Int, Float32}(lastspike = lastspike, q = q, τ = τ)

push!(synapse::Alpha, spike::Integer) = (synapse.lastspike = spike)

"""
    (synapse::Alpha)(t::Real; dt::Real = 1.0)

Evaluate an alpha synapse. See [`Synapse.Alpha`](@ref).
"""
function (synapse::Alpha)(t::Real; dt::Real = 1.0)
    @unpack lastspike, q, τ = synapse
    Δ = (t - lastspike) * dt
    v = Δ * (q / τ) * exp(-(Δ - τ) / τ)

    return (t >= lastspike) ? v : zero(v)
end

"""
    EPSP{T<:Real}

Synapse that returns `(ϵ₀ / τm - τs) * (exp(-Δ / τm) - exp(-Δ / τs)) Θ(Δ)`
(where `Θ` is the Heaviside function and `Δ = t - lastspike`).

Specifically, this is the EPSP time course for the SRM0 model introduced by Gerstner.
Details: [Spiking Neuron Models: Single Neurons, Populations, Plasticity]
         (https://icwww.epfl.ch/~gerstner/SPNM/node27.html#SECTION02323400000000000000)
"""
@kwdef struct EPSP{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::IT = 0
    ϵ₀::VT = 1
    τm::VT = 1
    τs::VT = 1
end

"""
    (synapse::EPSP)(t::Real; dt::Real = 1.0)

Evaluate an EPSP synapse. See [`Synapse.EPSP`](@ref).
"""
function (synapse::EPSP)(t::Real; dt::Real = 1.0)
    @unpack lastspike, ϵ₀, τm, τs = synapse
    Δ = dt * (t - lastspike)
    v = ϵ₀ / (τm - τs) * (exp(-Δ / τm) - exp(-Δ / τs))

    return (t >= lastspike) ? v : zero(v)
end

end