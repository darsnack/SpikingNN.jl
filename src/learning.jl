"""
    AbstractLearner

An abstract learning mechanism. Defined by required interface functions and whatever
data structures your learning mechanism needs.

Required interface functions:
- `prespike!(learner::AbstractLearner, w::Real, t::Integer, src_id::Integer, dest_id::Integer; dt::Real = 1.0)`:
    processes a pre-synaptic spike at time `t` along the synapse from `src_id` to `dest_id`
- `postspike!(learner::AbstractLearner, w::Real, t::Integer, src_id::Integer, dest_id::Integer; dt::Real = 1.0)`:
    processes a post-synaptic spike at time `t` along the synapse from `src_id` to `dest_id`
- `update!(learner::AbstractLearner, w::Real, t::Integer, src_id::Integer, dest_id::Integer; dt::Real = 1.0)`:
    return the updated weight along the synapse from `src_id` to `dest_id` since the last call to `update!()`

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
prespike!(learner::George, w, spikes; dt::Real = 1.0) = return
postspike!(learner::George, w, spikes; dt::Real = 1.0) = return
record!(learner::George, w, spikes; dt::Real = 1.0) = return
update!(learner::George, w, t::Integer; dt::Real = 1.0) = w

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
mutable struct STDP{T<:Real, VT<:AbstractArray{<:Real}} <: AbstractLearner
    A₊::Real
    A₋::Real
    τ₊::Real
    τ₋::Real
    lastpre::VT
    lastpost::VT
end

"""
    STDP(A₀::Real, τ::Real, n::Integer)

Create an STDP learner for `n` neuron population with weight change amplitude `A₀` and decay `τ`.
"""
STDP(A₀::Real, τ::Real, n::Integer) = STDP{Float32, Matrix{Float32}}(A₀, -A₀, τ, τ, zeros(n, n), zeros(n, n))

function prespike!(learner::STDP, w, spikes; dt::Real = 1.0)
    connectivity = (w .!= 0)
    pre = connectivity .* repeat(spikes, 1, size(w, 2))
    mask = (pre .> 0)
    learner.lastpre[mask] .= pre[mask]
end

function postspike!(learner::STDP, w, spikes; dt::Real = 1.0)
    connectivity = (w .!= 0)
    post = connectivity .* repeat(permutedims(spikes), size(w, 1), 1)
    mask = (post .> 0)
    learner.lastpost[mask] .= post[mask]
end

function record!(learner::STDP, w, spikes; dt::Real = 1.0)
    connectivity = (w .!= 0)
    pre = connectivity .* repeat(spikes, 1, length(spikes))
    mask = (pre .> 0)
    learner.lastpre[mask] .= pre[mask]
    post = connectivity .* repeat(permutedims(spikes), length(spikes), 1)
    mask = (post .> 0)
    learner.lastpost[mask] .= post[mask]
end

function update!(learner::STDP, w, t::Integer; dt::Real = 1.0)
    fpos = (x, y) -> (x > y && x == t) ? learner.A₊ * exp(-abs(x - y) * dt / learner.τ₊) : zero(eltype(w))
    fneg = (x, y) -> (x < y && y == t) ? learner.A₋ * exp(-abs(x - y) * dt / learner.τ₋) : zero(eltype(w))

    w .+= fpos.(learner.lastpost, learner.lastpre) .+ fneg.(learner.lastpost, learner.lastpre)
end