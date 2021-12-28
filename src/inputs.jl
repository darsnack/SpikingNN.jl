abstract type AbstractInput end

reset!(::AbstractInput) = nothing

"""
    ConstantRate(rate, rng, dims = (1,))
    ConstantRate(rate; rng = Random.GLOBAL_RNG, dims = (1,))
    ConstantRate(; freq, dt = 1, rng = Random.GLOBAL_RNG, dims = (1,))

Create a constant rate input where the probability a spike occurs is Bernoulli(`rate`).
Alternatively, specify `freq` in Hz at a simulation time step of `dt`.
"""
struct ConstantRate{T<:AbstractArray{<:Real}, S<:AbstractRNG} <: AbstractInput
    rate::T
    rng::S

    function ConstantRate(rate::T, rng::S) where {T<:AbstractArray{<:Real}, S<:AbstractRNG}
        @assert all(@. rate <= 1 || rate >= 0) "Cannot create a constant rate input for rate ∉ [0, 1] (supplied rate = $rate)."

        new{T, S}(rate, rng)
    end
end

ConstantRate(rate, rng, dims = (1,)) = ConstantRate(_fillmemaybe(rate, dims), rng)
ConstantRate(rate; rng = Random.GLOBAL_RNG, dims = (1,)) = ConstantRate(rate, rng, dims)
ConstantRate(; freq, dt = 1, rng = Random.GLOBAL_RNG, dims = (1,)) = ConstantRate(freq .* dt; rng = rng, dims = dims)

Base.size(input::ConstantRate) = size(input.rate)

"""
    evaluate!(spikes, input::ConstantRate, t; dt = 1)

Evaluate a constant rate-code input at time `t`.
"""
function evaluate!(spikes, input::ConstantRate{T}, t; dt = 1) where T
    r = adapt(T, rand(input.rng, size(input)...))
    @. spikes = ifelse(r < input.rate, t, zero(t))

    return spikes
end

"""
    StepCurrent(τ, dims = (1,))
    StepCurrent(; τ, dims = (1,))

Create a step current input that turns on at time `τ` seconds.
"""
struct StepCurrent{T<:AbstractArray{<:Real}} <: AbstractInput
    τ::T
end

StepCurrent(τ, dims = (1,)) = StepCurrent(_fillmemaybe(τ, dims))
StepCurrent(; τ, dims = (1,)) = StepCurrent(τ, dims)

Base.size(input::StepCurrent) = size(input.τ)

"""
    evaluate!(spikes, input::StepCurrent, t; dt = 1)

Evaluate a step current input at time `t`.
"""
function evaluate!(spikes, input::StepCurrent, t; dt = 1)
    @. spikes = ifelse(t * dt > input.τ, t, zero(t))

    return spikes
end

"""
    PoissonInput(λ, ρ₀, rng = Random.GLOBAL_RNG, dims = (1,))
    PoissonInput(; λ, ρ₀, rng = Random.GLOBAL_RNG, dims = (1,))

Create a inhomogenous Poisson input function according to

``X < \\mathrm{d}t \\rho_0 \\lambda(t)``

where ``X \\sim \\mathrm{Unif}([0, 1])``.

Fields:
- `ρ₀`: baseline firing rate
- `λ`: function (`(t::Integer; dt::Integer) -> Real`) that
       returns the instantaneous firing rate at time `t`

!!! warning
    If `λ` is not specified as an array, then `fill` is used to create an array of
    `λ` with size `dims`. When `λ` is a struct, this means every element references
    the same instance of `λ`.
"""
mutable struct PoissonInput{T<:AbstractArray{<:Real}, F<:AbstractArray, S<:AbstractRNG} <: AbstractInput
    ρ₀::T
    λ::F
    rng::S
end

PoissonInput(λ, ρ₀, rng = Random.GLOBAL_RNG, dims = (1,)) =
    PoissonInput(_fillmemaybe(ρ₀, dims), _fillmemaybe(λ, dims), rng)
PoissonInput(; λ, ρ₀, rng = Random.GLOBAL_RNG, dims = (1,)) = PoissonInput(λ, ρ₀, rng, dims)

"""
    evaluate!(input::PoissonInput, t::Integer; dt::Real = 1.0)
    (::PoissonInput)(t::Integer; dt::Real = 1.0)
    evaluate!(inputs::AbstractArray{<:PoissonInput}, t::Integer; dt::Real = 1.0)

Evaluate a inhomogenous Poisson input at time `t`.
"""
function evaluate!(spikes, input::PoissonInput{T}, t; dt = 1) where T
    d = adapt(T, map(λi -> λi(t; dt = dt), input.λ))
    r = adapt(T, rand(input.rng, size(input)...))
    @. spikes = ifelse(r < dt * input.ρ₀ * d, t, zero(t))

    return spikes
end

struct FunctionalInput{F<:AbstractArray, T<:AbstractRNG} <: AbstractInput
    f::F
    rng::T
end

FunctionalInput(f; rng = Random.GLOBAL_RNG, dims = (1,)) = FunctionalInput(_fillmemaybe(f, dims), rng)

Base.size(input::FunctionalInput) = size(input.f)

function evaluate!(spikes, input::FunctionalInput, t; dt = 1)
    r = rand(input.rng, size(input)...)
    f = map(fi -> fi(t; dt = dt), input.f)
    @. spikes = ifelse(r < f, t, zero(t))

    return spikes
end
