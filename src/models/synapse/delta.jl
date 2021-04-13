"""
    delta(t, q)
    delta!(I::AbstractArray, t::AbstractArray, q::AbstractArray)

Evaluate of a Dirac-delta synaptic response at time `t`.
Store the result directly in `I` if specified.

# Fields
- `I`: destination array for output current
- `t`: current time
- `q`: amplitude
"""
@inline delta(t, q) = (t ≈ 0) ? q : zero(q)
delta(t::AbstractArray, q::AbstractArray) = delta!(similar(q), t, q)
function delta!(I::AbstractArray, t::AbstractArray, q::AbstractArray)
    @. I = delta(t, q)

    return I
end

"""
    Delta{IT<:Integer, VT<:Real}
    Delta{IT, VT}(;q::Real = 1)
    Delta(;q::Real = 1)

A synapse representing a Dirac-delta at `lastspike` with amplitude `q`.
"""
mutable struct Delta{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::VT
    q::VT
end
Delta{IT, VT}(;q::Real = 1) where {IT<:Integer, VT<:Real} = Delta{IT, VT}(-Inf, q)
Delta(;q::Real = 1) = Delta{Int, Float32}(q = q)

"""
    excite!(synapse::Delta, spike::Integer)
    excite!(synapses::AbstractArray{<:Delta}, spike::Integer)

Excite `synapse` with a `spike` (`spike` == time step of spike).
"""
function excite!(synapse::Delta, spike::Integer)
    if spike > 0
        synapse.lastspike = spike
    end

    return synapse
end
function excite!(synapses::AbstractArray{<:Delta}, spike::Integer)
    if spike > 0
        synapses.lastspike .= spike
    end

    return synapses
end

# isactive(synapse::Delta, t::Integer; dt::Real = 1.0) = (t * dt == synapse.lastspike)
# isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Delta} = any(t * dt .== synapses.lastspike)

"""
    evaluate!(synapse::Delta, t::Integer; dt::Real = 1.0)
    (synapse::Delta)(t::Integer; dt::Real = 1.0)
    evaluate!(synapses::AbstractArray{<:Delta}, t::Integer; dt::Real = 1.0)
    evaluate!(current, synapses::AbstractArray{<:Delta}, t::Integer; dt::Real = 1.0)

Return `synapse.q` if `t == synapse.lastspike` otherwise return zero.
"""
evaluate!(synapse::Delta, t::Integer; dt::Real = 1.0) = delta((t - synapse.lastspike) * dt, synapse.q)
(synapse::Delta)(t::Integer; dt::Real = 1.0) = evaluate!(synapse, t; dt = dt)
evaluate!(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Delta} =
    delta((t .- synapses.lastspike) * dt, synapses.q)
evaluate!(current, synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Delta} =
    delta!(current, (t .- synapses.lastspike) * dt, synapses.q)

"""
    reset!(synapse::Delta)
    reset!(synapses::AbstractArray{<:Delta})

Reset `synapse`.
"""
function reset!(synapse::Delta)
    synapse.lastspike = -Inf

    return synapse
end
function reset!(synapses::AbstractArray{<:Delta})
    synapses.lastspike .= -Inf

    return synapses
end