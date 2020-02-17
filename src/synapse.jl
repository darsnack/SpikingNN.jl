module Synapse

using UnPack

"""
    SynapseParameters

A data structure to store the timing parameters of a synapse.

Fields:
- `Tw::Real`: the length of time that the response function is non-zero
    (i.e. the response will be sampled over the interval `[0, T_window]`)
- `dt::Real`: the sampling rate
- `samples::Vector{Real}`: an array to store samples (to be populated by `sampleresponse!()`)
"""
struct SynapseParameters{T<:Real}
    Tw::T
    dt::T
    samples::Vector{T}

    function SynapseParameters{T}(Tw, dt, samples::Vector{T}) where {T<:Real}
        (cld(Tw, dt) != length(samples)) && error("SynapseParameters requires ceil(Tw, dt) == length(samples) (received Tw = $Tw, dt = $dt, length = $(length(samples))")

        new{T}(Tw, dt, samples)
    end
end
SynapseParameters{T}(Tw::Real, dt::Real) where {T<:Real} = SynapseParameters{T}(Tw, dt, zeros(T, Int(cld(Tw, dt))))

"""
    AbstractSynapse

Inherit from this type to create a concrete synapse.

Expected Fields:
- `params::SynapseParameters`: the timing parameters of the response function
"""
abstract type AbstractSynapse end

"""
    sampleresponse(response)
    sampleresponse!(response)

Return the vector of samples representing the response function.
Call `sampleresponse!()` to resample the function.
"""
sampleresponse(response::AbstractSynapse) = response.params.samples, length(response.params.samples)
function sampleresponse!(response::AbstractSynapse)
    @unpack Tw, dt = response.params
    N = length(response.params.samples)
    t = collect(1:N)
    response.params.samples .= response.(dt .* t .- dt)

    return response.params.samples, N
end

"""
    Delta{T<:Real}

Synapse that returns `q` whenever `t = 0` and zero otherwise.
"""
struct Delta{T<:Real} <: AbstractSynapse
    params::SynapseParameters{T}
    q::T

    function Delta{T}(params::SynapseParameters{T}, q) where {T<:Real}
        response = new{T}(params, q)
        sampleresponse!(response)

        return response
    end
end

"""
    Delta(q::Real = 1; dt::Real = 1.0)

Create a Dirac delta synapse.
Optionally, specify `dt` to compute the appropriate `T_window`.
"""
Delta{T}(q::Real = 1; dt::Real = 1.0) where {T<:Real} = Delta{T}(SynapseParameters{T}(2 * dt, dt), q)
Delta(q::Real = 1; dt::Real = 1.0) = Delta{Float64}(q; dt = dt)

"""
    (::Delta)(Δ::Real)

Evaluate Dirac delta synapse.
"""
(synapse::Delta)(Δ::Real) = (Δ == 0) ? synapse.q : zero(synapse.q)

"""
    Alpha{T<:Real}

Synapse that returns `Δ * (q / τ) * exp(-(Δ - τ) / τ) Θ(Δ)`
(where `Θ` is the Heaviside function).
"""
struct Alpha{T<:Real} <: AbstractSynapse
    params::SynapseParameters{T}
    q::T
    τ::T

    function Alpha{T}(params::SynapseParameters{T}, q, τ) where {T<:Real}
        response = new{T}(params, q, τ)
        sampleresponse!(response)

        return response
    end
end

"""
    Alpha()

Create an alpha synapse.
"""
Alpha{T}(q::Real = 1, τ::Real = 1; dt::Real = 1.0) where {T<:Real} = Alpha{T}(SynapseParameters{T}(10 * τ, dt), q, τ)
Alpha(q::Real = 1, τ::Real = 1; dt::Real = 1.0) = Alpha{Float64}(q, τ; dt = dt)

"""
    (::Alpha)(Δ::Real)

Evaluate an alpha synapse.
"""
function (synapse::Alpha)(Δ::Real)
    @unpack q, τ = synapse
    v = Δ * (q / τ) * exp(-(Δ - τ) / τ)

    return (Δ >= 0) ? v : zero(v)
end

"""
    EPSP{T<:Real}

Synapse that returns `(ϵ₀ / τm - τs) * (exp(-Δ / τm) - exp(-Δ / τs)) Θ(Δ)`
(where `Θ` is the Heaviside function).

Specifically, this is the EPSP time course for the SRM0 model with an
alpha synapse.
Details: https://icwww.epfl.ch/~gerstner/SPNM/node27.html#SECTION02323400000000000000
"""
struct EPSP{T<:Real} <: AbstractSynapse
    params::SynapseParameters{T}
    ϵ₀::T
    τm::T
    τs::T

    function EPSP{T}(params::SynapseParameters{T}, ϵ₀, τm, τs) where {T<:Real}
        response = new{T}(params, ϵ₀, τm, τs)
        sampleresponse!(response)

        return response
    end
end

"""
    EPSP()

Create an EPSP synapse.
"""
EPSP{T}(ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1; dt::Real = 1.0) where {T<:Real} = EPSP{T}(SynapseParameters{T}(τs + 8 * τm, dt), ϵ₀, τm, τs)
EPSP(ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1; dt::Real = 1.0) = EPSP{Float64}(ϵ₀, τm, τs; dt = dt)

"""
    (::EPSP)(Δ::Real)

Evaluate an EPSP synapse.
"""
function (synapse::EPSP)(Δ::Real)
    @unpack ϵ₀, τm, τs = synapse
    v = ϵ₀ / (τm - τs) * (exp(-Δ / τm) - exp(-Δ / τs))

    return (Δ >= 0) ? v : zero(v)
end

end