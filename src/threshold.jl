module Threshold

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
struct Poisson{T<:Real}
    dt::T
    ρ₀::T
    Θ::T
    Δᵤ::T
end

"""
    (::Poisson)(Δ::Real, v::Real)

Evaluate Poisson threshold function.

Fields:
- `Δ::Real`: time difference (typically `t - last_spike_out`)
- `v::Real`: current membrane potential
"""
function (threshold::Poisson)(Δ::Real, v::Real)
    ρ = threshold.ρ₀ * exp((v - threshold.Θ) / threshold.Δᵤ)
    x = rand()
    return x < ρ * threshold.dt
end

end