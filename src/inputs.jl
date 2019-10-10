"""
    ConstantRate(rate::Real)

Create a constant rate input where the probability a spike occurs is Bernoulli(rate).
rate-coded neuron firing at a fixed rate.
"""
struct ConstantRate
    dist::Bernoulli
    rate::Real
    function ConstantRate(rate::Real)
        (rate > 1 || rate < 0) && error("Cannot create a constant rate input for rate ∉ [0, 1] (supplied rate = $rate).")
        dist = Bernoulli(rate)
        new(dist, rate)
    end
end

"""
    (::ConstantRate)(t::Integer; dt::Real = 1.0)

Evaluate a constant rate-code input at time `t`.
Optionally, specify `dt` if the simulation timestep is not 1.0.
"""
(input::ConstantRate)(t::Integer; dt::Real = 1.0) = rand(input.dist)

"""
    StepCurrent(τ::Real)

Create a step current input that turns on at time `τ` seconds.
"""
struct StepCurrent
    τ::Real
end

"""
    (::StepCurrent)(t::Integer; dt::Real = 1.0)

Evaluate a step current input at time `t`.
Optionally, specify `dt` if the simulation timestep is not 1.0.
"""
(input::StepCurrent)(t::Integer; dt::Real = 1.0) = (t * dt > input.τ) ? 1 : 0

"""
    PoissonInput(ρ₀::Real, σ::Real, x, x₀; metric = (x, y) -> sum((x .- y).^2))

Create a inhomogenous Poisson input function according to

``X < \\mathrm{d}t \\rho_0 \\exp\\left(-\\frac{d_{\\text{metric}}(x, x_0)}{\\sigma^2}\\right)``

where ``X \\sim \\mathrm{Unif}([0, 1])``.
Note that `dt` **must** be appropriately specified to ensure correct behavior.

Fields:
- `ρ₀::Real`: baseline firing rate
- `σ::Real`: separation deviation
- `x₀`: baseline comparison
- `metric::(Real, Real) -> Real`: distance metric for comparison
"""
mutable struct PoissonInput
    ρ₀::Real
    σ::Real
    x
    x₀
    metric
    PoissonInput(ρ₀::Real, σ::Real, x, x₀; metric = (x, y) -> sum((x .- y).^2)) = new(ρ₀, σ, x, x₀, metric)
end

"""
    (::PoissonInput)(t::Integer; dt::Real = 1.0)

Evaluate a inhomogenous Poisson input at time `t`.
Optionally, specify `dt` if the simulation time step is not 1.0.
"""
(input::PoissonInput)(t::Integer; dt::Real = 1.0) =
    (rand() < dt * input.ρ₀ * exp(-input.metric(input.x, input.x₀) / input.σ^2)) ? 1 : 0