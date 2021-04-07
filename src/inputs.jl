abstract type AbstractInput end

# isactive(input::AbstractInput, t::Integer) = true

"""
    ConstantRate(rate::Real)
    ConstantRate{T}(rate::Real)
    ConstantRate(freq::Real, dt::Real)

Create a constant rate input where the probability a spike occurs is Bernoulli(`rate`).
rate-coded neuron firing at a fixed rate.
Alternatively, specify `freq` in Hz at a simulation time step of `dt`.
"""
struct ConstantRate{T<:Real, RT<:AbstractRNG} <: AbstractInput
    dist::Bernoulli
    rate::T
    rng::RT
end
function ConstantRate{T}(rate::Real; rng::RT = Random.GLOBAL_RNG) where {T<:Real, RT}
    if rate > 1 || rate < 0
        error("Cannot create a constant rate input for rate ∉ [0, 1] (supplied rate = $rate).")
    end

    dist = Bernoulli(rate)
    ConstantRate{T, RT}(dist, rate, rng)
end
ConstantRate(rate::Real; kwargs...) = ConstantRate{Float32}(rate; kwargs...)
ConstantRate(freq::Real, dt::Real; kwargs...) = ConstantRate(freq * dt; kwargs...)

"""
    evaluate!(input::ConstantRate, t::Integer; dt::Real = 1.0)
    (::ConstantRate)(t::Integer; dt::Real = 1.0)
    evaluate!(inputs::AbstractArray{<:ConstantRate}, t::Integer; dt::Real = 1.0)

Evaluate a constant rate-code input at time `t`.
"""
evaluate!(input::ConstantRate, t::Integer; dt::Real = 1.0) =
    (rand(input.rng, input.dist) == 1) ? t : zero(t)
(input::ConstantRate)(t::Real; dt::Real = 1.0) = evaluate!(input, t; dt = dt)
evaluate!(inputs::AbstractArray{<:ConstantRate}, t::I; dt::Real = 1.0) where I<:Integer =
    evaluate!(Array{I}(undef, size(inputs)...), inputs, t; dt = dt)
function evaluate!(spikes, inputs::AbstractArray{<:ConstantRate}, t::Integer; dt::Real = 1.0)
    spikes .= ifelse.(rand.(inputs.rng, inputs.dist) .== 1, t, zero(t))

    return spikes
end

"""
    StepCurrent(τ::Real)

Create a step current input that turns on at time `τ` seconds.
"""
struct StepCurrent{T<:Real} <: AbstractInput
    τ::T
end

"""
    evaluate!(input::StepCurrent, t::Integer; dt::Real = 1.0)
    (::StepCurrent)(t::Integer; dt::Real = 1.0)
    evaluate!(inputs::AbstractArray{<:StepCurrent}, t::Integer; dt::Real = 1.0)

Evaluate a step current input at time `t`.
"""
evaluate!(input::StepCurrent, t::Integer; dt::Real = 1.0) = (t * dt > input.τ) ? t : zero(t)
(input::StepCurrent)(t::Integer; dt::Real = 1.0) = evaluate!(input, t; dt = dt)
evaluate!(inputs::AbstractArray{<:StepCurrent}, t::I; dt::Real = 1.0) where I<:Integer =
    evaluate!(Array{I}(undef, size(inputs)...), inputs, t; dt = dt)
function evaluate!(spikes, inputs::AbstractArray{<:StepCurrent}, t::Integer; dt::Real = 1.0)
    spikes .= ifelse.(t * dt .> inputs.τ, t, zero(t))

    return spikes
end

"""
    PoissonInput(ρ₀::Real, λ::Function)

Create a inhomogenous Poisson input function according to

``X < \\mathrm{d}t \\rho_0 \\lambda(t)``

where ``X \\sim \\mathrm{Unif}([0, 1])``.
Note that `dt` **must** be appropriately specified to ensure correct behavior.

Fields:
- `ρ₀::Real`: baseline firing rate
- `λ::(Integer; dt::Integer) -> Real`: a function that returns the
    instantaneous firing rate at time `t`
"""
mutable struct PoissonInput{T<:Real, F, RT<:AbstractRNG} <: AbstractInput
    ρ₀::T
    λ::F
    rng::RT
end
PoissonInput{T}(ρ₀::Real, λ::F; rng::RT = Random.GLOBAL_RNG) where {T<:Real, F, RT} =
    PoissonInput{T, F, RT}(ρ₀, λ, rng)
PoissonInput(args...; kwargs...) = PoissonInput{Float32}(args...; kwargs...)

"""
    evaluate!(input::PoissonInput, t::Integer; dt::Real = 1.0)
    (::PoissonInput)(t::Integer; dt::Real = 1.0)
    evaluate!(inputs::AbstractArray{<:PoissonInput}, t::Integer; dt::Real = 1.0)

Evaluate a inhomogenous Poisson input at time `t`.
"""
evaluate!(input::PoissonInput, t::Integer; dt::Real = 1.0) =
    (rand(input.rng) < dt * input.ρ₀ * input.λ(t; dt = dt)) ? t : zero(t)
(input::PoissonInput)(t::Integer; dt::Real = 1.0) = evaluate!(input, t; dt = dt)
evaluate!(inputs::AbstractArray{<:PoissonInput}, t::I; dt::Real = 1.0) where I<:Integer =
    evaluate!(Array{I}(undef, size(inputs)...), inputs, t; dt = dt)
function evaluate!(spikes, inputs::AbstractArray{<:PoissonInput}, t::Integer; dt::Real = 1.0)
    ρ₀ = inputs.ρ₀
    d = adapt(typeof(ρ₀), [λ(t; dt = dt) for λ in inputs.λ])
    r = adapt(typeof(ρ₀), rand.(inputs.rng))
    @. spikes = ifelse(r < dt * ρ₀ * d, t, zero(t))

    return spikes
end

"""
    InputPopulation{IT<:StructArray{<:AbstractInput}}

An `InputPopulation` is a population of `AbstractInput`s.
"""
struct InputPopulation{IT<:StructArray{<:AbstractInput}}
    inputs::IT
end
InputPopulation(inputs::AbstractArray) = InputPopulation(StructArray(inputs))

"""
    size(pop::InputPopulation)

Return the number of neurons in a population.
"""
Base.size(pop::InputPopulation) = length(pop.inputs)

Base.IndexStyle(::Type{<:InputPopulation}) = IndexLinear()
Base.getindex(pop::InputPopulation, i::Int) = pop.inputs[i]
function Base.setindex!(pop::InputPopulation, input::AbstractInput, i::Int)
    pop.inputs[i] = input
end

Base.show(io::IO, pop::InputPopulation) = print(io, "Population{$(eltype(pop.inputs))}($(size(pop)))")
Base.show(io::IO, ::MIME"text/plain", pop::InputPopulation) = show(io, pop)

"""
    evaluate!(pop::InputPopulation, t::Integer; dt::Real = 1.0)
    (::InputPopulation)(t::Integer; dt::Real = 1.0)

Evaluate a population of inputs at time `t`.
"""
evaluate!(pop::InputPopulation, t::Integer; dt::Real = 1.0) = evaluate!(pop.inputs, t; dt = dt)
evaluate!(spikes, pop::InputPopulation, t; dt = 1.0) = evaluate!(spikes, pop.inputs, t; dt = dt)
(pop::InputPopulation)(t::Integer; dt::Real = 1.0) = evaluate!(pop, t; dt = dt)