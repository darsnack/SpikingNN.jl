"""
    constant_rate(rate::Real, n::Integer)

Create an array (of length `n`) of spike times corresponding to a
rate-coded neuron firing at a fixed rate.
"""
function constant_rate(rate::Real, n::Integer; response = delta)
    if n < 1
        error("Cannot create a spike train of length < 1 (supplied n = $n).")
    elseif rate < 0
        error("Cannot create a spike train with rate < 0 (supplied rate = $rate).")
    end

    dist = Bernoulli(rate)
    spikes = rand(dist, n)
    times = findall(x -> x == 1, spikes)
    return times
end

"""
    step_current(τ::Real, T::Real; dt::Real = 1.0)

Create an array of spike times corresponding to a step-input current
that turns on at time `τ` and lasts `T` seconds.
Optionally, specify `dt` if the simulation timestep is not 1.0.
"""
function step_current(τ::Real, T::Real; dt::Real = 1.0)
    if T < τ
        error("Cannot create a step current with transition time, τ = $τ, later than total time, T = $T.")
    end

    N = Int(T / dt) + 1
    n = Int(ceil(τ / dt)) + 1
    return collect(n:N)
end

"""
    poissoninput(ρ₀::Real, Θ::Real; dt::Real)

Create a inhomogenous Poisson input function according to

``X < \\mathrm{d}t \\rho_0 \\exp\\left(\\frac{d_{\\text{metric}}(x, x_{\\text{base}})}{\\sigma^2}\\right)``

where ``X \\sim \\mathrm{Unif}([0, 1])``.
Note that `dt` **must** be appropriately specified to ensure correct behavior.

Fields:
- `ρ₀::Real`: baseline firing rate
- `xbase::AbstractArray`: baseline comparison
- `σ::Real`: separation deviation
- `dt::Real`: simulation time step
- `metric::(Real, Real) -> Real`: distance metric for comparison
"""
poissoninput(ρ₀::Real, xbase::AbstractArray, σ::Real; dt::Real, metric = (x, y) -> norm(x .- y)^2) =
    x -> ρ₀ * exp(metric(x, xbase) / σ^2)