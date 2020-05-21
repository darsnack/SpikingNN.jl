abstract type AbstractInput end

"""
    isactive(input::AbstractInput, t::Integer)

Return true (inputs are always active).
"""
isactive(input::AbstractInput, t::Integer) = true

"""
    isdone(input::AbstractInput)

Return true (inputs are never done).
"""
isdone(input::AbstractInput) = true

"""
    ConstantRate(rate::Real)

Create a constant rate input where the probability a spike occurs is Bernoulli(`rate`).
rate-coded neuron firing at a fixed rate.
"""
struct ConstantRate{T<:Real} <: AbstractInput
    dist::Bernoulli
    rate::T
end
function ConstantRate{T}(rate::Real) where {T<:Real}
    (rate > 1 || rate < 0) && error("Cannot create a constant rate input for rate ∉ [0, 1] (supplied rate = $rate).")
    dist = Bernoulli(rate)
    ConstantRate{T}(dist, rate)
end
ConstantRate(rate::Real) = ConstantRate{Float32}(rate)
ConstantRate(freq::Real, dt::Real) = ConstantRate(freq * dt)

"""
    (::ConstantRate)(t::Integer; dt::Real = 1.0)

Evaluate a constant rate-code input at time `t`.
Optionally, specify `dt` if the simulation timestep is not 1.0.
"""
(input::ConstantRate)(t::Integer; dt::Real = 1.0) = (rand(input.dist) == 1) ? t : zero(t)
evalinputs(inputs::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:ConstantRate} =
    (rand.(inputs.dist) .== 1) .* adapt(typeof(inputs.rate), fill(t, size(inputs.rate)))

"""
    StepCurrent(τ::Real)

Create a step current input that turns on at time `τ` seconds.
"""
struct StepCurrent{T<:Real} <: AbstractInput
    τ::T
end

"""
    (::StepCurrent)(t::Integer; dt::Real = 1.0)

Evaluate a step current input at time `t`.
Optionally, specify `dt` if the simulation timestep is not 1.0.
"""
(input::StepCurrent)(t::Integer; dt::Real = 1.0) = (t * dt > input.τ) ? t : zero(t)
evalinputs(inputs::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:StepCurrent} =
    (t * dt .> inputs.τ) .* t

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
mutable struct PoissonInput{T<:Real, F} <: AbstractInput
    ρ₀::T
    λ::F
end

"""
    (::PoissonInput)(t::Integer; dt::Real = 1.0)

Evaluate a inhomogenous Poisson input at time `t`.
Optionally, specify `dt` if the simulation time step is not 1.0.
"""
(input::PoissonInput)(t::Integer; dt::Real = 1.0) =
    (rand() < dt * input.ρ₀ * input.λ(t; dt = dt)) ? t : zero(t)
function evalinputs(inputs::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:PoissonInput}
    ρ₀ = inputs.ρ₀
    d = adapt(typeof(ρ₀), [λ(t; dt = dt) for λ in inputs.λ])
    r = adapt(typeof(ρ₀), rand(length(inputs)))

    return (r .< dt .* ρ₀ .* d) .* t
end

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

(pop::InputPopulation)(t::Integer; dt::Real = 1.0) = evalinputs(pop.inputs, t; dt = dt)