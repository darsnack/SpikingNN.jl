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
                  rng = Random.GLOBAL_RNG)
    rho = @avx @. baserate * exp((v - theta) / deltav) * dt
    spikes .= rand(rng, size(rho)...) .< rho

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
    Poisson(ρ₀, θ, Δv, rng = Random.GLOBAL_RNG, dims = (1,))
    Poisson(; ρ₀, θ, Δv, rng = Random.GLOBAL_RNG, dims = (1,))

Choose to output a spike based on a inhomogenous Poisson process given by

``X < \\mathrm{d}t \\: \\rho_0 \\exp\\left(\\frac{v - \\Theta}{\\Delta_v}\\right)``

where ``X \\sim \\mathrm{Unif}([0, 1])``.
Accordingly, `dt` must be set correctly so that the neuron does not always spike.

Fields:
- `ρ₀::Real`: baseline firing rate at threshold
- `Θ::Real`: firing threshold
- `Δv::Real`: voltage resolution
- `rng`: random number generation
"""
struct Poisson{T<:AbstractArray{<:Real}, S<:AbstractRNG} <: AbstractThreshold
    ρ₀::T
    Θ::T
    Δv::T
    rng::S
end

Poisson(ρ₀, θ, Δv, rng = Random.GLOBAL_RNG, dims = (1,)) =
    Poisson(_fillmemaybe(ρ₀, dims), _fillmemaybe(θ, dims), _fillmemaybe(Δv, dims), rng)
Poisson(; ρ₀, θ, Δv, rng = Random.GLOBAL_RNG, dims = (1,)) = Poisson(ρ₀, θ, Δv, rng, dims)

Base.size(threshold::Poisson) = size(threshold.ρ₀)

"""
    evaluate!(threshold::Poisson, t::Integer, v::Real; dt::Real = 1.0)
    evaluate!(spikes, thresholds::AbstractArray{<:Poisson}, t::Integer, v; dt::Real = 1.0)

Evaluate Poisson threshold function. See [`Threshold.Poisson`](@ref).
"""
function evaluate!(spikes, threshold::Poisson, t, voltage; dt = 1)
    poisson!(spikes, threshold.ρ₀, threshold.Θ, threshold.Δv, voltage, dt, threshold.rng)
    @. spikes = ifelse(spikes, t, zero(t))

    return spikes
end
