@reexport module Threshold

using SNNlib.Threshold: poisson
using Adapt

import ..SpikingNN: excite!, reset!, isactive

export AbstractThreshold, isactive

abstract type AbstractThreshold end

"""
    Ideal(vth::Real)
"""
struct Ideal{T<:Real} <: AbstractThreshold
    vth::T
end

isactive(threshold::Ideal, t::Integer; dt::Real = 1.0) = false

(threshold::Ideal)(t::Real, v::Real; dt::Real = 1.0) = (v >= threshold.vth) ? t : zero(t)
evalthresholds(thresholds::T, t::Integer, v; dt::Real = 1.0) where T<:AbstractArray{<:Ideal} =
    (v .>= thresholds.vth) .* adapt(typeof(v), fill(t, size(v)))

"""
    Poisson(dt::Real, ρ₀::Real = 60, Θ::Real = 0.016, Δᵤ::Real = 0.002)

Choose to output a spike based on a inhomogenous Poisson process given by

``X < \\mathrm{d}t \\rho_0 \\exp\\left(\\frac{v - \\Theta}{\\Delta_u}\\right)``

where ``X \\sim \\mathrm{Unif}([0, 1])``.
Accordingly, `dt` must be set correctly so that the neuron does not always spike.

Fields:
- `dt::Real`: simulation time step (**must be set appropriately**)
- `ρ₀::Real`: baseline firing rate at threshold
- `Θ::Real`: firing threshold
- `Δᵤ::Real`: firing width
"""
struct Poisson{T<:Real} <: AbstractThreshold
    ρ₀::T
    Θ::T
    Δᵤ::T
end

"""
    isactive(threshold::Poisson, t)

Return true if threshold function will produce output at time step `t`.
"""
isactive(threshold::Poisson, t::Integer; dt::Real = 1.0) = true

"""
    (::Poisson)(Δ::Real, v::Real)

Evaluate Poisson threshold function.

Fields:
- `Δ::Real`: time difference (typically `t - last_spike_out`)
- `v::Real`: current membrane potential
"""
(threshold::Poisson)(t::Real, v::Real; dt::Real = 1.0) =
    poisson(threshold.ρ₀, threshold.Θ, threshold.Δᵤ, v; dt = dt) ? t : zero(t)
evalthresholds(thresholds::T, t::Integer, v; dt::Real = 1.0) where T<:AbstractArray{<:Poisson} =
    poisson(thresholds.ρ₀, thresholds.Θ, thresholds.Δᵤ, v; dt = dt) .* adapt(typeof(v), fill(t, size(v)))


end