"""
    AbstractLearner

An abstract learning mechanism. Defined by required interface functions and whatever
data structures your learning mechanism needs.

Required interface functions:
- `prespike!(learner::AbstractLearner, w::Real, t::Integer, src_id::Integer, dest_id::Integer; dt::Real = 1.0)`:
    processes a pre-synaptic spike at time `t` along the synapse from `src_id` to `dest_id`
- `postspike!(learner::AbstractLearner, w::Real, t::Integer, src_id::Integer, dest_id::Integer; dt::Real = 1.0)`:
    processes a post-synaptic spike at time `t` along the synapse from `src_id` to `dest_id`
- `update!(learner::AbstractLearner, src_id::Integer, dest_id::Integer)`: return the weight
    change from along the synapse from `src_id` to `dest_id` since the last call to `update!()`

This could be, for example, Hebbian learning. The functions `prespike!()` and `postspike!()`
are called whenever the pre-synaptic neuron or post-synaptic neuron fires, respectively.
When each function is called, it processes the spike according to the learning rule, and
updates an internal weight change state, `Δw`.
When `update!()` is called, the current `Δw` is returned then reset to zero.
"""
abstract type AbstractLearner end

"""
    George

A dumb learner that does nothing when processing spikes.
Aptly named after my dog, George (https://darsnack.github.io/website/about)
"""
struct George <: AbstractLearner end
prespike!(learner::George, w::Real, t::Integer, src_id::Integer, dest_id::Integer; dt::Real = 1.0) = return
postspike!(learner::George, w::Real, t::Integer, src_id::Integer, dest_id::Integer; dt::Real = 1.0) = return
update!(learner::George, src_id::Integer, dest_id::Integer) = 0

"""
    STDP

A simple spike-timing-dependent plasticity mechanism.

Fields:
- `A₊::Real`: positive weight change amplitude
- `A₋::Real`: negative weight change amplitude
- `τ₊::Real`: positive decay parameter
- `τ₋::Real`: negative decay parameter
- `lastpre::Array{Float64, 2}`: the last occurence of a pre-synaptic spike
    (matrix row = src, col = dest)
- `lastpost::Array{Float64, 2}`: the last occurence of a post-synaptic spike
    (matrix row = src, col = dest)
- `Δw::Array{Float64, 2}`: the current synaptic weight change
    (matrix row = src, col = dest)
"""
mutable struct STDP <: AbstractLearner
    A₊::Real
    A₋::Real
    τ₊::Real
    τ₋::Real
    lastpre::Array{Float64, 2}
    lastpost::Array{Float64, 2}
    Δw::Array{Float64, 2}
end

"""
    STDP(A₀::Real, τ::Real, n::Integer)

Create an STDP learner for `n` neuron population with weight change amplitude `A₀` and decay `τ`.
"""
STDP(A₀::Real, τ::Real, n::Integer) = STDP(A₀, -A₀, τ, τ, zeros(n, n), zeros(n, n), zeros(n, n))

function prespike!(learner::STDP, w::Real, t::Integer, src_id::Integer, dest_id::Integer; dt::Real = 1.0)
    Δt = learner.lastpost[src_id, dest_id] - t * dt
    learner.Δw[src_id, dest_id] += (Δt < 0) ? learner.A₋ * exp(-abs(Δt) / learner.τ₋) : zero(w)
    learner.lastpre[src_id, dest_id] = t * dt
end

function postspike!(learner::STDP, w::Real, t::Integer, src_id::Integer, dest_id::Integer; dt::Real = 1.0)
    Δt = t * dt - learner.lastpre[src_id, dest_id]
    learner.Δw[src_id, dest_id] += (Δt > 0) ? learner.A₊ * exp(-abs(Δt) / learner.τ₊) : zero(w)
    learner.lastpost[src_id, dest_id] = t * dt
end

function update!(learner::STDP, src_id::Integer, dest_id::Integer)
    Δw = learner.Δw[src_id, dest_id]
    learner.Δw[src_id, dest_id] = 0

    return Δw
end