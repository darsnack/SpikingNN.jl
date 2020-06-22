@reexport module Synapse

using SpikingNNFunctions.Synapse: delta, alpha, epsp
using DataStructures: Queue, enqueue!, dequeue!, empty!
using DataStructures: CircularBuffer, fill!, push!, empty!
using Adapt

import ..SpikingNN: excite!, spike!, evaluate!, reset!, isactive

export AbstractSynapse, QueuedSynapse, DelayedSynapse,
       excite!, spike!, reset!, isactive

"""
    AbstractSynapse

Inherit from this type to create a concrete synapse.
"""
abstract type AbstractSynapse end

"""
    excite!(synapse::AbstractSynapse, spikes::Vector{<:Integer})
    excite!(synapses::AbstractArray{<:AbstractSynapse}, spikes::Vector{<:Integer})

Excite a `synapse` with a vector of spikes by calling `excite!(synapse, spike) for spike in spikes`.
"""
excite!(synapse::AbstractSynapse, spikes::Vector{<:Integer}) = map(x -> excite!(synapse, x), spikes)
excite!(synapses::AbstractArray{<:AbstractSynapse}, spikes::Vector{<:Integer}) = map(x -> excite!(synapses, x), spikes)

"""
    spike!(synapse::AbstractSynapse, spike::Integer; dt::Real = 1.0)
    spike!(synapse::AbstractArray{<:AbstractSynapse}, spikes::AbstractArray{<:Integer}; dt::Real = 1.0)

Notify a synapse that the post-synaptic neuron has released a spike.
The default implmentation is to do nothing. Override this behavior by dispatching on your synapse type.
"""
spike!(synapse::AbstractSynapse, spike::Integer; dt::Real = 1.0) = nothing
spike!(synapses::AbstractArray{<:AbstractSynapse}, spikes; dt::Real = 1.0) = nothing



"""
    QueuedSynapse{ST<:AbstractSynapse, IT<:Integer}

A `QueuedSynapse` excites its internal synapse when the timestep matches the head of the queue.
Wrapping a synapse in this type allows you to pre-load several spike excitation times, and the
  internal synapse will be excited as those time stamps are evaluated.
This can be useful for cases where it is more efficient to load all the input spikes before simulation.

*Note: currently only supported on CPU.*
"""
struct QueuedSynapse{ST<:AbstractSynapse, IT<:Integer} <: AbstractSynapse
    core::ST
    queue::Queue{IT}
end
QueuedSynapse{IT}(synapse) where {IT<:Integer} = QueuedSynapse{typeof(synapse), IT}(synapse, Queue{IT}())
QueuedSynapse(synapse) = QueuedSynapse{typeof(synapse), Int}(synapse, Queue{Int}())

_ispending(queue, t) = !isempty(queue) && first(queue) <= t
function _shiftspike!(queue, lastspike, t)
    while _ispending(queue, t)
        lastspike = dequeue!(queue)
    end

    return lastspike
end
function _shiftspike!(queues::AbstractArray, lastspikes, t)
    pending = map(x -> _ispending(x, t), queues)
    while any(pending)
        @. lastspikes[pending] = dequeue!(queues[pending])
        pending = map(x -> _ispending(x, t), queues)
    end

    return lastspikes
end

"""
    excite!(synapse::QueuedSynapse, spike::Integer)
    excite!(synapses::AbstractArray{<:QueuedSynapse}, spike::Integer)

Excite `synapse` with a `spike` (`spike` == time step of spike) by pushing
  `spike` onto `synapse.queue`.
"""
excite!(synapse::QueuedSynapse, spike::Integer) = enqueue!(synapse.queue, spike)
excite!(synapses::T, spike::Integer) where T<:AbstractArray{<:QueuedSynapse} =
    map(x -> enqueue!(x, spike), synapses.queue)

isactive(synapse::QueuedSynapse, t::Integer; dt::Real = 1.0) = _ispending(synapse.queue, t) || isactive(synapse.core, t; dt = dt)
isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:QueuedSynapse} =
    any(map(x -> _ispending(x, t), synapses.queue)) || isactive(synapses.core, t; dt = dt)

"""
    evaluate!(synapse::QueuedSynapse, t::Integer; dt::Real = 1.0)
    (synapse::QueuedSynapse)(t::Integer; dt::Real = 1.0)
    evaluate!(synapses::AbstractArray{<:QueuedSynapse}, t::Integer; dt::Real = 1.0)

Evaluate `synapse` at time `t` by first exciting `synapse.core` with a spike if
  there is one to process, then evaluating `synapse.core`.
"""
function evaluate!(synapse::QueuedSynapse, t::Integer; dt::Real = 1.0)
    excite!(synapse.core, _shiftspike!(synapse.queue, 0, t))

    return synapse.core(t; dt = dt)
end
(synapse::QueuedSynapse)(t::Integer; dt::Real = 1.0) = evaluate!(synapse, t; dt = dt)
function evaluate!(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:QueuedSynapse}
    lastspikes = _shiftspike!(synapses.queue, zeros(Int, size(synapses)), t)
    @inbounds for i in eachindex(synapses)
        excite!(view(synapses.core, i), lastspikes[i])
    end

    return evaluate!(synapses.core, t; dt = dt)
end

"""
    reset!(synapse::QueuedSynapse)
    reset!(synapses::AbstractArray{<:QueuedSynapse})

Clear `synapse.queue` and reset `synapse.core`.
"""
function reset!(synapse::QueuedSynapse)
    empty!(synapse.queue)
    reset!(synapse.core)
end
function reset!(synapses::T) where T<:AbstractArray{<:QueuedSynapse}
    empty!.(synapses.queue)
    reset!(synapses.core)
end



"""
    DelayedSynapse

A `DelayedSynapse` adds a fixed delay to spikes when exciting its internal synapse.
"""
struct DelayedSynapse{T<:Real, ST<:AbstractSynapse} <: AbstractSynapse
    core::ST
    delay::T
end

"""
    excite!(synapse::DelayedSynapse, spike::Integer)
    excite!(synapses::AbstractArray{<:DelayedSynapse}, spike::Integer)

Excite `synapse.core` with a `spike` + `synapse.delay` (`spike` == time step of spike).
"""
excite!(synapse::DelayedSynapse, spike::Integer) = excite!(synapse.core, spike + synapse.delay)
function excite!(synapses::T, spike::Integer) where T<:AbstractArray{<:DelayedSynapse}
    delayedspikes = adapt(Array{eltype(synapses.delay), ndims(synapses)}, spike .+ synapses.delay)
    if spike > 0
        @inbounds for i in eachindex(synapses)
            excite!(view(synapses.core, i), delayedspikes[i])
        end
    end
end

isactive(synapse::DelayedSynapse, t::Integer; dt::Real = 1.0) = isactive(synapse.core, t; dt = dt)
isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:DelayedSynapse} =
    isactive(synapses.core, t; dt = dt)

"""
    evaluate!(synapse::DelayedSynapse, t::Integer; dt::Real = 1.0)
    (synapse::DelayedSynapse)(t::Integer; dt::Real = 1.0)
    evaluate!(synapses::AbstractArray{<:DelayedSynapse}, t::Integer; dt::Real = 1.0)

Evaluate `synapse.core` at time `t`.
"""
evaluate!(synapse::DelayedSynapse, t::Integer; dt::Real = 1.0) = synapse.core(t; dt = dt)
(synapse::DelayedSynapse)(t::Integer; dt::Real = 1.0) = evaluate!(synapse, t; dt = dt)
evaluate!(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:DelayedSynapse} =
    evaluate!(synapses.core, t; dt = dt)

"""
    reset!(synapse::DelayedSynapse)
    reset!(synapses::AbstractArray{<:DelayedSynapse})

Reset `synapse.core`.
"""
reset!(synapse::DelayedSynapse) = reset!(synapse.core)
reset!(synapses::T) where T<:AbstractArray{<:DelayedSynapse} = reset!(synapses.core)



"""
    Delta{IT<:Integer, VT<:Real}

A synapse representing a Dirac-delta at `lastspike` with amplitude `q`.
"""
mutable struct Delta{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::VT
    q::VT
end

"""
    Delta{IT, VT}(;q::Real = 1)
    Delta(;q::Real = 1)

Create a Dirac-delta synapse with amplitude `q`.
"""
Delta{IT, VT}(;q::Real = 1) where {IT<:Integer, VT<:Real} = Delta{IT, VT}(-Inf, q)
Delta(;q::Real = 1) = Delta{Int, Float32}(q = q)

"""
    excite!(synapse::Delta, spike::Integer)
    excite!(synapses::AbstractArray{<:Delta}, spike::Integer)

Excite `synapse` with a `spike` (`spike` == time step of spike).
"""
excite!(synapse::Delta, spike::Integer) = (spike > 0) && (synapse.lastspike = spike)
excite!(synapses::T, spike::Integer) where T<:AbstractArray{<:Delta} = (spike > 0) && (synapses.lastspike .= spike)

isactive(synapse::Delta, t::Integer; dt::Real = 1.0) = (t * dt == synapse.lastspike)
isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Delta} = any(t * dt .== synapses.lastspike)

"""
    evaluate!(synapse::Delta, t::Integer; dt::Real = 1.0)
    (synapse::Delta)(t::Integer; dt::Real = 1.0)
    evaluate!(synapses::AbstractArray{<:Delta}, t::Integer; dt::Real = 1.0)

Return `synapse.q` if `t == synapse.lastspike` otherwise return zero.
"""
evaluate!(synapse::Delta, t::Integer; dt::Real = 1.0) = delta(t * dt, synapse.lastspike * dt, synapse.q)
(synapse::Delta)(t::Integer; dt::Real = 1.0) = evaluate!(synapse, t; dt = dt)
evaluate!(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Delta} =
    delta(t * dt, synapses.lastspike * dt, synapses.q)

"""
    reset!(synapse::Delta)
    reset!(synapses::AbstractArray{<:Delta})

Reset `synapse`.
"""
reset!(synapse::Delta) = (synapse.lastspike = -Inf)
reset!(synapses::T) where T<:AbstractArray{<:Delta} = (synapses.lastspike .= -Inf)



"""
    Alpha{IT<:Integer, VT<:Real}

Synapse that returns `(t - lastspike) * (q / τ) * exp(-(t - lastspike - τ) / τ) Θ(t - lastspike)`
(where `Θ` is the Heaviside function).
"""
mutable struct Alpha{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::VT
    q::VT
    τ::VT
end

"""
    Alpha{IT, VT}(;q::Real = 1, τ::Real = 1)
    Alpha(;q::Real = 1, τ::Real = 1)

Create an alpha synapse with amplitude `q` and time constant `τ`.
"""
Alpha{IT, VT}(;q::Real = 1, τ::Real = 1) where {IT<:Integer, VT<:Real} = Alpha{IT, VT}(-Inf, q, τ)
Alpha(;q::Real = 1, τ::Real = 1) = Alpha{Int, Float32}(q = q, τ = τ)

"""
    excite!(synapse::Alpha, spike::Integer)
    excite!(synapses::AbstractArray{<:Alpha}, spike::Integer)

Excite `synapse` with a `spike` (`spike` == time step of spike).
"""
excite!(synapse::Alpha, spike::Integer) = (spike > 0) && (synapse.lastspike = spike)
excite!(synapses::T, spike::Integer) where T<:AbstractArray{<:Alpha} = (spike > 0) && (synapses.lastspike .= spike)

isactive(synapse::Alpha, t::Real; dt::Real = 1.0) = dt * (t - synapse.lastspike) <= 10 * synapse.τ
isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Alpha} =
    any(dt .* (t .- synapses.lastspike) .<= 10 .* synapses.τ)

"""
    evaluate!(synapse::Alpha, t::Integer; dt::Real = 1.0)
    (synapse::Alpha)(t::Integer; dt::Real = 1.0)
    evaluate!(synapses::AbstractArray{<:Alpha}, t::Integer; dt::Real = 1.0)

Evaluate an alpha synapse. See [`Synapse.Alpha`](@ref).
"""
evaluate!(synapse::Alpha, t::Integer; dt::Real = 1.0) = alpha(t * dt, synapse.lastspike * dt, synapse.q, synapse.τ)
(synapse::Alpha)(t::Integer; dt::Real = 1.0) = evaluate!(synapse, t; dt = dt)
evaluate!(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Alpha} =
    alpha(t * dt, synapses.lastspike * dt, synapses.q, synapses.τ)

"""
    reset!(synapse::Alpha)
    reset!(synapses::AbstractArray{<:Alpha})

Reset `synapse`.
"""
reset!(synapse::Alpha) = (synapse.lastspike = -Inf)
reset!(synapses::T) where T<:AbstractArray{<:Alpha}= (synapses.lastspike .= -Inf)



"""
    EPSP{T<:Real}

Synapse that returns `(ϵ₀ / τm - τs) * (exp(-Δ / τm) - exp(-Δ / τs)) Θ(Δ)`
(where `Θ` is the Heaviside function and `Δ = t - lastspike`).

Specifically, this is the EPSP time course for the SRM0 model introduced by Gerstner.
Details: [Spiking Neuron Models: Single Neurons, Populations, Plasticity]
         (https://icwww.epfl.ch/~gerstner/SPNM/node27.html#SECTION02323400000000000000)
"""
mutable struct EPSP{IT<:Integer, VT<:Real} <: AbstractSynapse
    spikes::CircularBuffer{VT}
    ϵ₀::VT
    τm::VT
    τs::VT
end

"""
    EPSP{IT, VT}(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1, N = 100)
    EPSP(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1, N = 100)

Create an EPSP synapse with amplitude `ϵ₀`, rise time `τs`, and fall time `τm`.
Specify `N` to adjust how many pre-synaptic spikes are remembered between post-synaptic spikes.
"""
EPSP{IT, VT}(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1, N = 100) where {IT<:Integer, VT<:Real} =
    EPSP{IT, VT}(fill!(CircularBuffer{VT}(N), -Inf), ϵ₀, τm, τs)
EPSP(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1, N = 100) = EPSP{Int, Float32}(ϵ₀ = ϵ₀, τm = τm, τs = τs, N = N)

"""
    excite!(synapse::EPSP, spike::Integer)
    excite!(synapses::AbstractArray{<:EPSP}, spike::Integer)

Excite `synapse` with a `spike` (`spike` == time step of spike).
"""
excite!(synapse::EPSP, spike::Integer) = (spike > 0) && push!(synapse.spikes, spike)
excite!(synapses::T, spike::Integer) where T<:AbstractArray{<:EPSP} = (spike > 0) && push!.(synapses.spikes, spike)

"""
    spike!(synapse::EPSP, spike::Integer; dt::Real = 1.0)
    spike!(synapses::AbstractArray{<:EPSP}, spikes; dt::Real = 1.0)

Reset `synapse` when the post-synaptic neuron spikes.
"""
spike!(synapse::EPSP, spike::Integer; dt::Real = 1.0) = reset!(synapse)
spike!(synapses::T, spikes; dt::Real = 1.0) where T<:AbstractArray{<:EPSP} = reset!(synapses)

isactive(synapse::EPSP, t::Integer; dt::Real) = dt * (t - first(synapse.spikes)) <= synapse.τs + 8 * synapse.τm
isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:EPSP} =
    any(dt .* (t .- first.(synapses.spikes)) .<= synapses.τs .+ 8 .* synapses.τm)

"""
    evaluate!(synapse::EPSP, t::Integer; dt::Real = 1.0)
    (synapse::EPSP)(t::Integer; dt::Real = 1.0)
    evaluate!(synapses::AbstractArray{<:EPSP}, t::Integer; dt::Real = 1.0)

Evaluate an EPSP synapse. See [`Synapse.EPSP`](@ref).
"""
evaluate!(synapse::EPSP, t::Integer; dt::Real = 1.0) =
    mapreduce(tf -> epsp(t * dt, tf * dt, synapse.ϵ₀, synapse.τm, synapse.τs), +, synapse.spikes)
(synapse::EPSP)(t::Integer; dt::Real = 1.0) = evaluate!(synapse, t; dt = dt)
function evaluate!(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:EPSP}
    N = length(synapses.spikes[1])
    return mapreduce(i -> epsp(t * dt, adapt(typeof(synapses.ϵ₀), getindex.(synapses.spikes, i) * dt), synapses.ϵ₀, synapses.τm, synapses.τs), +, 1:N)
end

"""
    reset!(synapse::EPSP)
    reset!(synapses::AbstractArray{<:EPSP})

Reset `synapse`.
"""
reset!(synapse::EPSP) = fill!(empty!(synapse.spikes), -Inf)
reset!(synapses::T) where T<:AbstractArray{<:EPSP}= fill!.(empty!.(synapses.spikes), -Inf)

end