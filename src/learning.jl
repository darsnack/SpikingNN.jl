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
prespike!(learner::George, src, dst, w, spikes; dt::Real = 1.0) = return
postspike!(learner::George, src, dst, w, spikes; dt::Real = 1.0) = return
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
struct STDP{T<:Real, VT<:AbstractArray{<:Real}} <: AbstractLearner
    A₊::T
    A₋::T
    τ₊::T
    τ₋::T
    lastpre::VT
    lastpost::VT
end

"""
    STDP(A₀::Real, τ::Real, n::Integer)

Create an STDP learner for `n` neuron population with weight change amplitude `A₀` and decay `τ`.
"""
STDP(A₀::Real, τ::Real, n::Integer) = STDP{Float32, Matrix{Float32}}(A₀, -A₀, τ, τ, zeros(n, n), zeros(n, n))

function prespike!(learner::STDP, w, spikes; dt::Real = 1.0)
    f(x, y, w) = (w != 0) && (y > 0) ? y : x
    @cast learner.lastpre[i, j] = f(learner.lastpre[i, j], spikes[i], w[i, j])
end

function postspike!(learner::STDP, w, spikes; dt::Real = 1.0)
    f(x, y, w) = (w != 0) && (y > 0) ? y : x
    @cast learner.lastpost[i, j] = f(learner.lastpost[i, j], spikes[j], w[i, j])
end

_stdpfpos(A, τ, t, dt, x, y) = (x > y && x == t) * A * exp(-abs(x - y) * dt / τ)
_stdpfneg(A, τ, t, dt, x, y) = (x < y && y == t) * A * exp(-abs(x - y) * dt / τ)

function update!(learner::STDP, w, t::Integer; dt::Real = 1.0)
    A₊, A₋, τ₊, τ₋ = learner.A₊, learner.A₋, learner.τ₊, learner.τ₋
    map!((x, y, w) -> w + _stdpfpos(A₊, τ₊, t, dt, x, y) + _stdpfneg(A₋, τ₋, t, dt, x, y),
         w, learner.lastpost, learner.lastpre, w)
end