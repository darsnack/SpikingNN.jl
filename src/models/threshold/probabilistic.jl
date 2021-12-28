"""
    Probabilistic(f; rng = Random.GLOBAL_RNG, dims = (1,))

A probabilistic threshold function produces a spike with probability `f(voltage)`.

!!! warning
    If `f` is not specified as an array, then `fill` is used to create an array of
    `f` with size `dims`. When `f` is a struct, this means every element references
    the same instance of `f`.
"""
struct Probabilistic{F<:AbstractArray, T<:AbstractRNG} <: AbstractThreshold
    f::F
    rng::T
end

Probabilistic(f; rng = Random.GLOBAL_RNG, dims = (1,)) = Probabilistic(_fillmemaybe(f, dims), rng)

Base.size(threshold::Probabilistic) = size(threshold.f)

function evaluate!(spikes, threshold::Probabilistic, t, voltage; dt = 1)
    r = rand(threshold.rng, size(threshold)...)
    @. spikes = ifelse((r < map(threshold.f, voltage)), t, zero(t))

    return spikes
end
