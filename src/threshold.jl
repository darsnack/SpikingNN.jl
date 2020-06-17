@reexport module Threshold

using SpikingNNFunctions.Threshold: poisson
using Adapt

import ..SpikingNN: excite!, reset!, isactive

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

(threshold::Ideal)(t::Real, v::Real; dt::Real = 1.0) = (v >= threshold.vth) ? t : zero(t)
evalthresholds(thresholds::T, t::Integer, v; dt::Real = 1.0) where T<:AbstractArray{<:Ideal} =
    Int.((v .>= thresholds.vth) .* adapt(typeof(v), fill(t, size(v))))

"""
    Poisson(ρ₀::Real, Θ::Real, Δᵤ::Real)

Choose to output a spike based on a inhomogenous Poisson process given by

``X < \\mathrm{d}t \\rho_0 \\exp\\left(\\frac{v - \\Theta}{\\Delta_u}\\right)``

where ``X \\sim \\mathrm{Unif}([0, 1])``.
Accordingly, `dt` must be set correctly so that the neuron does not always spike.

Fields:
- `ρ₀::Real`: baseline firing rate at threshold
- `Θ::Real`: firing threshold
- `Δᵤ::Real`: voltage resolution
"""
struct Poisson{T<:Real} <: AbstractThreshold
    ρ₀::T
    Θ::T
    Δᵤ::T
end

isactive(threshold::Poisson, t::Integer; dt::Real = 1.0) = true

"""
    (::Poisson)(t::Integer, v::Real; dt::Real = 1.0)
    evalthresholds(thresholds::AbstractArray{<:Poisson}, t::Integer, v; dt::Real = 1.0)

Evaluate Poisson threshold function. See [`Threshold.Poisson`](@ref).
"""
(threshold::Poisson)(t::Integer, v::Real; dt::Real = 1.0) =
    poisson(threshold.ρ₀, threshold.Θ, threshold.Δᵤ, v; dt = dt) ? t : zero(t)
evalthresholds(thresholds::T, t::Integer, v; dt::Real = 1.0) where T<:AbstractArray{<:Poisson} =
    Int.(poisson(thresholds.ρ₀, thresholds.Θ, thresholds.Δᵤ, v; dt = dt) .* adapt(typeof(v), fill(t, size(v))))

end