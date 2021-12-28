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
    Delta(q, dims = (1,))
    Delta(; q = one(Float32), dims = (1,))

A Dirac-Delta synaptic array of size `dims` with amplitude `q`.
"""
struct Delta{T<:AbstractArray{<:Real}} <: AbstractSynapse
    offset::T
    q::T
end

Delta(q, dims = (1,)) = Delta(fill(-inf(eltype(q)), dims), _fillmemaybe(q, dims))
Delta(; q = one(Float32), dims = (1,)) = Delta(q, dims)

Base.getindex(synapse::Delta, I...) = Delta(synapse.offset[I...], synapse.q[I...])
function Base.setindex!(synapse::Delta, v::Delta, I...)
    synapse.offset[I...] .= v.offset[I...]
    synapse.q[I...] .= v.q[I...]

    return synapse
end
Base.size(synapse::Delta) = size(synapse.offset)

"""
    excite!(synapse::Delta, spike::Integer)

Excite `synapse` with a `spike` (`spike` == time step of spike).
"""
function excite!(synapse::Delta, spikes)
    spiked = spikes .> 0
    synapse.offset[spiked] .= spikes[spiked]

    return synapse
end

# isactive(synapse::Delta, t::Integer; dt::Real = 1.0) = (t * dt == synapse.lastspike)
# isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Delta} = any(t * dt .== synapses.lastspike)

"""
    evaluate!(synapse::Delta, t; dt = 1)
    evaluate!(current, synapses::Delta, t; dt = 1)

Return `synapse.q` if `t == synapse.offset` otherwise return zero.
"""
evaluate!(synapse::Delta, t; dt = 1) = delta((t .- synapse.offset) .* dt, synapse.q)
evaluate!(current, synapse::Delta, t; dt = 1) = delta!(current, (t .- synapse.offset) .* dt, synapse.q)

"""
    reset!(synapse::Delta)

Reset `synapse` by setting the last pre-synaptic spike time to `-Inf`.
"""
function reset!(synapse::Delta, mask = trues(size(synapse)))
    synapse.offset[mask] .= -Inf

    return synapse
end
