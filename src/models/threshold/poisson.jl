"""
    poisson(baserate, theta, deltav, v, dt, rng::AbstractRNG)
    poisson(baserate::AbstractArray,
            theta::AbstractArray,
            deltav::AbstractArray,
            v::AbstractArray,
            dt,
            rng::AbstractRNG)
    poisson!(spikes::AbstractArray,
             baserate::AbstractArray,
             theta::AbstractArray,
             deltav::AbstractArray,
             v::AbstractArray,
             dt,
             rng::AbstractRNG)

Evaluate inhomogeneous Poisson process threshold functions.
Modeled as

``X < \\mathrm{d}t \\rho_0 \\exp\\left(\\frac{v - \\Theta}{\\Delta_u}\\right)``

where ``X \\sim \\mathrm{Unif}([0, 1])``.

Optionally store results into `spikes`.

# Fields
- `spikes`: 
- `baserate`: base line firing rate
- `theta`: threshold potential
- `deltav`: potential resolution
- `v`: current membrane potential
- `dt`: simulation timestep
- `rng`: random number generator (ignored for `CuArray` inputs)
"""
function poisson(baserate, theta, deltav, v, dt, rng::AbstractRNG = Random.GLOBAL_RNG)
    rho = baserate * exp((v - theta) / deltav)

    return rand(rng) < rho * dt
end
function poisson!(spikes::AbstractArray,
                  baserate::AbstractArray,
                  theta::AbstractArray,
                  deltav::AbstractArray,
                  v::AbstractArray,
                  dt,
                  rng = fill(Random.GLOBAL_RNG, size(v)))
    rho = @avx @. baserate * exp((v - theta) / deltav) * dt
    spikes .= rand.(rng) .< rho

    return spikes
end
poisson(baserate::AbstractArray,
        theta::AbstractArray,
        deltav::AbstractArray,
        v::AbstractArray,
        dt,
        rng = fill(Random.GLOBAL_RNG, size(v))) =
    poisson!(similar(v), baserate, theta, deltav, v, dt, rng)
function poisson!(spikes::CuArray,
                  baserate::CuArray,
                  theta::CuArray,
                  deltav::CuArray,
                  v::CuArray,
                  dt,
                  rng = Random.GLOBAL_RNG)
    rho = @. baserate * exp((v - theta) / deltav) * dt

    return CUDA.rand(size(rho)...) .< rho
end

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
struct Poisson{T<:Real, RT<:AbstractRNG} <: AbstractThreshold
    ρ₀::T
    Θ::T
    Δᵤ::T
    rng::RT
end

Poisson{T}(;ρ₀::Real, Θ::Real, Δᵤ::Real, rng::RT = Random.GLOBAL_RNG) where {T<:Real, RT} =
    Poisson{T, RT}(ρ₀, Θ, Δᵤ, rng)
Poisson(;kwargs...) = Poisson{Float32}(;kwargs...)

isactive(threshold::Poisson, t::Integer; dt::Real = 1.0) = true

"""
    evaluate!(threshold::Poisson, t::Integer, v::Real; dt::Real = 1.0)
    (::Poisson)(t::Integer, v::Real; dt::Real = 1.0)
    evaluate!(thresholds::AbstractArray{<:Poisson}, t::Integer, v; dt::Real = 1.0)
    evaluate!(spikes, thresholds::AbstractArray{<:Poisson}, t::Integer, v; dt::Real = 1.0)

Evaluate Poisson threshold function. See [`Threshold.Poisson`](@ref).
"""
evaluate!(threshold::Poisson, t::Integer, v::Real; dt::Real = 1.0) =
    poisson(threshold.ρ₀, threshold.Θ, threshold.Δᵤ, v, dt, threshold.rng) ? t : zero(t)
(threshold::Poisson)(t::Integer, v::Real; dt::Real = 1.0) = evaluate!(threshold, t, v; dt = dt)
function evaluate!(thresholds::T, t::I, v; dt::Real = 1.0) where {T<:AbstractArray{<:Poisson}, I<:Integer}
    spikes = poisson(thresholds.ρ₀, thresholds.Θ, thresholds.Δᵤ, v, dt, thresholds.rng)
    
    return I.(spikes .* t)
end
function evaluate!(spikes, thresholds::T, t::I, v; dt::Real = 1.0) where {T<:AbstractArray{<:Poisson}, I<:Integer}
    poisson!(spikes, thresholds.ρ₀, thresholds.Θ, thresholds.Δᵤ, v, dt, thresholds.rng)
    spikes .= I.(spikes .* t)
    
    return spikes
end
