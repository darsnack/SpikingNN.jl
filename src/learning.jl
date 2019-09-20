"""
    AbstractLearner

An abstract learning mechanism. Defined by required interface functions and whatever
data structures your learning mechanism needs.

Required interface functions:
- `prespike(learner::AbstractLearner, w<:Real, t<:Real, src_id<:Integer, dest_id<:Integer)`:
  processes a pre-synaptic spike at time `t` along the synapse from `src_id` to `dest_id`
  (returns a weight change)
- `postspike(learner::AbstractLearner, w<:Real, t<:Real, src_id<:Integer, dest_id<:Integer)`:
  processes a post-synaptic spike at time `t` along the synapse from `src_id` to `dest_id`
  (returns a weight change)

This could be, for example, Hebbian learning. The functions `prespike()` and `postspike()`
are called whenever the pre-synaptic neuron or post-synaptic neuron fires, respectively.
When each function is called, it processes the spike according to the learning rule,
and it returns a weight change (return zero for no change).
"""
abstract type AbstractLearner end

"""
    George

A dumb learner that does nothing when processing spikes.
Aptly named after my dog, George (https://darsnack.github.io/website/about)
"""
struct George <: AbstractLearner end
prespike(learner::George, w::Real, t::Real, src_id::Integer, dest_id::Integer) = 0
postspike(learner::George, w::Real, t::Real, src_id::Integer, dest_id::Integer) = 0

"""
    STDP

A simple spike-timing-dependent plasticity mechanism.

Fields:
- `A₊::Real`: positive weight change amplitude
- `A₋::Real`: negative weight change amplitude
- `τ₊::Real`: positive decay parameter
- `τ₋::Real`: negative decay parameter
- `lastpre::Array{Real, 2}`: the last occurence of a pre-synaptic spike
  (matrix row = src, col = dest)
- `lastpost::Array{Real, 2}`: the last occurence of a post-synaptic spike
  (matrix row = src, col = dest)
"""
mutable struct STDP <: AbstractLearner
    A₊::Real
    A₋::Real
    τ₊::Real
    τ₋::Real
    lastpre::Array{Float64, 2}
    lastpost::Array{Float64, 2}
end

"""
    STDP(A₀::Real, τ::Real, n::Integer)

Create an STDP learner for `n` neuron population with weight change amplitude `A₀` and decay `τ`.
"""
STDP(A₀::Real, τ::Real, n::Integer) = STDP(A₀, -A₀, τ, τ, zeros(n, n), zeros(n, n))

function prespike(learner::STDP, w::Real, t::Real, src_id::Integer, dest_id::Integer)
    Δt = learner.lastpost[src_id, dest_id] - t
    Δw = (Δt < 0) ? learner.A₋ * exp(-abs(Δt) / learner.τ₋) : zero(w)
    learner.lastpre[src_id, dest_id] = t

    return Δw
end

function postspike(learner::STDP, w::Real, t::Real, src_id::Integer, dest_id::Integer)
    Δt = t - learner.lastpre[src_id, dest_id]
    Δw = (Δt > 0) ? learner.A₊ * exp(-abs(Δt) / learner.τ₊) : zero(w)
    learner.lastpost[src_id, dest_id] = t

    return Δw
end