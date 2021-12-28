"""
    AbstractSynapse

Inherit from this type to create a concrete synapse.
"""
abstract type AbstractSynapse end

# """
#     excite!(synapse::AbstractSynapse, spikes::Vector{<:Integer})
#     excite!(synapses::AbstractArray{<:AbstractSynapse}, spikes::Vector{<:Integer})

# Excite a `synapse` with a vector of spikes by calling `excite!(synapse, spike) for spike in spikes`.
# """
# excite!(synapse::AbstractSynapse, spikes::Vector{<:Integer}) =
#     map(x -> excite!(synapse, x), spikes)
# excite!(synapses::AbstractArray{<:AbstractSynapse}, spikes::Vector{<:Integer}) =
#     map(x -> excite!(synapses, x), spikes)


# """
#     QueuedSynapse{ST<:AbstractSynapse, IT<:Integer}

# A `QueuedSynapse` excites its internal synapse when the timestep matches the head of the queue.
# Wrapping a synapse in this type allows you to pre-load several spike excitation times, and the
#   internal synapse will be excited as those time stamps are evaluated.
# This can be useful for cases where it is more efficient to load all the input spikes before simulation.

# *Note: currently only supported on CPU.*
# """
# struct QueuedSynapse{ST<:AbstractSynapse, IT<:Integer} <: AbstractSynapse
#     core::ST
#     queue::Queue{IT}
# end
# QueuedSynapse{IT}(synapse) where {IT<:Integer} = QueuedSynapse{typeof(synapse), IT}(synapse, Queue{IT}())
# QueuedSynapse(synapse) = QueuedSynapse{typeof(synapse), Int}(synapse, Queue{Int}())

# _ispending(queue, t) = !isempty(queue) && first(queue) <= t
# function _shiftspike!(queue, lastspike, t)
#     while _ispending(queue, t)
#         lastspike = dequeue!(queue)
#     end

#     return lastspike
# end
# function _shiftspike!(queues::AbstractArray, lastspikes, t)
#     pending = map(x -> _ispending(x, t), queues)
#     while any(pending)
#         @. lastspikes[pending] = dequeue!(queues[pending])
#         pending = map(x -> _ispending(x, t), queues)
#     end

#     return lastspikes
# end

# """
#     excite!(synapse::QueuedSynapse, spike::Integer)
#     excite!(synapses::AbstractArray{<:QueuedSynapse}, spike::Integer)

# Excite `synapse` with a `spike` (`spike` == time step of spike) by pushing
#   `spike` onto `synapse.queue`.
# """
# excite!(synapse::QueuedSynapse, spike::Integer) = enqueue!(synapse.queue, spike)
# excite!(synapses::T, spike::Integer) where T<:AbstractArray{<:QueuedSynapse} =
#     map(x -> enqueue!(x, spike), synapses.queue)

# isactive(synapse::QueuedSynapse, t::Integer; dt::Real = 1.0) = _ispending(synapse.queue, t) || isactive(synapse.core, t; dt = dt)
# isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:QueuedSynapse} =
#     any(map(x -> _ispending(x, t), synapses.queue)) || isactive(synapses.core, t; dt = dt)

# """
#     evaluate!(synapse::QueuedSynapse, t::Integer; dt::Real = 1.0)
#     (synapse::QueuedSynapse)(t::Integer; dt::Real = 1.0)
#     evaluate!(synapses::AbstractArray{<:QueuedSynapse}, t::Integer; dt::Real = 1.0)

# Evaluate `synapse` at time `t` by first exciting `synapse.core` with a spike if
#   there is one to process, then evaluating `synapse.core`.
# """
# function evaluate!(synapse::QueuedSynapse, t::Integer; dt::Real = 1.0)
#     excite!(synapse.core, _shiftspike!(synapse.queue, 0, t))

#     return synapse.core(t; dt = dt)
# end
# (synapse::QueuedSynapse)(t::Integer; dt::Real = 1.0) = evaluate!(synapse, t; dt = dt)
# function evaluate!(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:QueuedSynapse}
#     lastspikes = _shiftspike!(synapses.queue, zeros(Int, size(synapses)), t)
#     @inbounds for i in eachindex(synapses)
#         excite!(view(synapses.core, i), lastspikes[i])
#     end

#     return evaluate!(synapses.core, t; dt = dt)
# end

# """
#     reset!(synapse::QueuedSynapse)
#     reset!(synapses::AbstractArray{<:QueuedSynapse})

# Clear `synapse.queue` and reset `synapse.core`.
# """
# function reset!(synapse::QueuedSynapse)
#     empty!(synapse.queue)
#     reset!(synapse.core)
# end
# function reset!(synapses::T) where T<:AbstractArray{<:QueuedSynapse}
#     empty!.(synapses.queue)
#     reset!(synapses.core)
# end


# """
#     DelayedSynapse

# A `DelayedSynapse` adds a fixed delay to spikes when exciting its internal synapse.
# """
# struct DelayedSynapse{T<:Real, ST<:AbstractSynapse} <: AbstractSynapse
#     core::ST
#     delay::T
# end

# """
#     excite!(synapse::DelayedSynapse, spike::Integer)
#     excite!(synapses::AbstractArray{<:DelayedSynapse}, spike::Integer)

# Excite `synapse.core` with a `spike` + `synapse.delay` (`spike` == time step of spike).
# """
# excite!(synapse::DelayedSynapse, spike::Integer) = excite!(synapse.core, spike + synapse.delay)
# function excite!(synapses::T, spike::Integer) where T<:AbstractArray{<:DelayedSynapse}
#     delayedspikes = adapt(Array{eltype(synapses.delay), ndims(synapses)}, spike .+ synapses.delay)
#     if spike > 0
#         @inbounds for i in eachindex(synapses)
#             excite!(view(synapses.core, i), delayedspikes[i])
#         end
#     end
# end

# isactive(synapse::DelayedSynapse, t::Integer; dt::Real = 1.0) = isactive(synapse.core, t; dt = dt)
# isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:DelayedSynapse} =
#     isactive(synapses.core, t; dt = dt)

# """
#     evaluate!(synapse::DelayedSynapse, t::Integer; dt::Real = 1.0)
#     (synapse::DelayedSynapse)(t::Integer; dt::Real = 1.0)
#     evaluate!(synapses::AbstractArray{<:DelayedSynapse}, t::Integer; dt::Real = 1.0)

# Evaluate `synapse.core` at time `t`.
# """
# evaluate!(synapse::DelayedSynapse, t::Integer; dt::Real = 1.0) = evaluate!(synapse.core, t; dt = dt)
# (synapse::DelayedSynapse)(t::Integer; dt::Real = 1.0) = evaluate!(synapse, t; dt = dt)
# evaluate!(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:DelayedSynapse} =
#     evaluate!(synapses.core, t; dt = dt)
# evaluate!(current, synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:DelayedSynapse} =
#     evaluate!(current, synapses.core, t; dt = dt)

# """
#     reset!(synapse::DelayedSynapse)
#     reset!(synapses::AbstractArray{<:DelayedSynapse})

# Reset `synapse.core`.
# """
# reset!(synapse::DelayedSynapse) = reset!(synapse.core)
# reset!(synapses::T) where T<:AbstractArray{<:DelayedSynapse} = reset!(synapses.core)
