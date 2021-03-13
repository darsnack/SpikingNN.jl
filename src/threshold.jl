@reexport module Threshold

using Adapt
using Random
using CUDA

import ..SpikingNN: excite!, evaluate!, reset!, isactive

export AbstractThreshold, isactive

abstract type AbstractThreshold end

"""
    Ideal(vth::Real)

An ideal threshold spikes when `v > vth`.
"""
struct Ideal{T<:Real} <: AbstractThreshold
    vth::T
end

isactive(threshold::Ideal, t::Integer; dt::Real = 1.0) = false

"""
    evaluate!(threshold::Ideal, t::Real, v::Real; dt::Real = 1.0)
    (threshold::Ideal)(t::Real, v::Real; dt::Real = 1.0)
    evaluate!(thresholds::AbstractArray{<:Ideal}, t::Integer, v; dt::Real = 1.0)

Return `t` when `v > threshold.vth`.
"""
evaluate!(threshold::Ideal, t::Real, v::Real; dt::Real = 1.0) = (v >= threshold.vth) ? t : zero(t)
(threshold::Ideal)(t::Real, v::Real; dt::Real = 1.0) = evaluate!(threshold, t, v; dt = dt)
evaluate!(thresholds::T, t::Integer, v; dt::Real = 1.0) where T<:AbstractArray{<:Ideal} =
    Int.((v .>= thresholds.vth) .* adapt(typeof(v), fill(t, size(v))))

"""
    Poisson(ρ₀::Real, Θ::Real, Δᵤ::Real, rng::AbstractRNG)

Choose to output a spike based on a inhomogenous Poisson process given by

``X < \\mathrm{d}t \\: \\rho_0 \\exp\\left(\\frac{v - \\Theta}{\\Delta_u}\\right)``

where ``X \\sim \\mathrm{Unif}([0, 1])``.
Accordingly, `dt` must be set correctly so that the neuron does not always spike.

Fields:
- `ρ₀::Real`: baseline firing rate at threshold
- `Θ::Real`: firing threshold
- `Δᵤ::Real`: voltage resolution
- `rng`: random number generation
"""
struct Poisson{T<:Real,  RT<:AbstractRNG} <: AbstractThreshold
    ρ₀::T
    Θ::T
    Δᵤ::T
    rng::RT
end

function Poisson{T}(ρ₀::Real, Θ::Real, Δᵤ::Real; rng::RT = Random.GLOBAL_RNG) where {T<:Real, RT}
    Poisson{T,RT}(ρ₀, Θ, Δᵤ, rng)
end
Poisson(ρ₀::Real, Θ::Real, Δᵤ::Real; kwargs...) = Poisson{Real}(ρ₀, Θ, Δᵤ; kwargs...)

isactive(threshold::Poisson, t::Integer; dt::Real = 1.0) = true

"""
    poisson(baserate, theta, deltav, v; dt::Real, rng::AbstractRNG)
    poisson(baserate::AbstractArray{<:Real}, theta::AbstractArray{<:Real}, deltav::AbstractArray{<:Real}, v::AbstractArray{<:Real}; dt::Real, rng::AbstractRNG)
    poisson(baserate::CuVector{<:Real}, theta::CuVector{<:Real}, deltav::CuVector{<:Real}, v::CuVector{<:Real}; dt::Real, rng::AbstractRNG)

Evaluate inhomogeneous Poisson process threshold functions.
Modeled as

``X < \\mathrm{d}t \\rho_0 \\exp\\left(\\frac{v - \\Theta}{\\Delta_u}\\right)``

where ``X \\sim \\mathrm{Unif}([0, 1])``.

Use `CuVector` instead of `Vector` to evaluate on GPU.

# Fields
- `baserate`: base line firing rate
- `theta`: threshold potential
- `deltav`: potential resolution
- `v`: current membrane potential
- `dt`: simulation timestep
- `rng`: random number generation
"""
function poisson(baserate, theta, deltav, v; dt::Real, rng::AbstractRNG = Random.GLOBAL_RNG)
    rho = baserate * exp((v - theta) / deltav)

    return rand(rng) < rho * dt
end
function poisson(baserate::AbstractArray{<:Real}, theta::AbstractArray{<:Real}, deltav::AbstractArray{<:Real}, v::AbstractArray{<:Real}; dt::Real, rng::AbstractRNG = Random.GLOBAL_RNG)
    rho = baserate .* exp.((v .- theta) ./ deltav)

    return rand(rng, length(rho)) .< rho .* dt
end
function poisson(baserate::CuVector{<:Real}, theta::CuVector{<:Real}, deltav::CuVector{<:Real}, v::CuVector{<:Real}; dt::Real, rng::AbstractRNG = Random.GLOBAL_RNG)
    rho = baserate .* exp.((v .- theta) ./ deltav)

    return CuArrays.rand(rng, length(rho)) .< rho .* dt
end

"""
    evaluate!(threshold::Poisson, t::Integer, v::Real; dt::Real = 1.0)
    (::Poisson)(t::Integer, v::Real; dt::Real = 1.0)
    evaluate!(thresholds::AbstractArray{<:Poisson}, t::Integer, v; dt::Real = 1.0)

Evaluate Poisson threshold function. See [`Threshold.Poisson`](@ref).
"""
evaluate!(threshold::Poisson, t::Integer, v::Real; dt::Real = 1.0) =
    poisson(threshold.ρ₀, threshold.Θ, threshold.Δᵤ, v; dt = dt, rng = threshold.rng) ? t : zero(t)
(threshold::Poisson)(t::Integer, v::Real; dt::Real = 1.0) = evaluate!(threshold, t, v; dt = dt)
evaluate!(thresholds::T, t::Integer, v; dt::Real = 1.0) where T<:AbstractArray{<:Poisson} =
    Int.(poisson(thresholds.ρ₀, thresholds.Θ, thresholds.Δᵤ, v; dt = dt, rng = first(thresholds.rng)) .* adapt(typeof(v), fill(t, size(v))))

end