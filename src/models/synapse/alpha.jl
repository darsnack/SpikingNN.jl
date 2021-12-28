"""
    alpha(t, q, tau)
    alpha(t::AbstractArray, q::AbstractArray, tau::AbstractArray)
    alpha!(I::AbstractArray, t::AbstractArray, q::AbstractArray, tau::AbstractArray)

Evaluate an alpha synaptic response at time `t`.
Modeled as `t * (q / τ) * exp(-(t - τ) / τ) Θ(t)`
(where `Θ` is the Heaviside function).
Store the output current in `I` when specified.

# Fields
- `I`: destination array for output current
- `t`: current time
- `q`: amplitude
- `tau`: time constant
"""
alpha(t, q, tau) = (t >= 0 && t < Inf) ? t * (q / tau) * exp(-(t - tau) / tau) : zero(q)
alpha(t::AbstractArray, q::AbstractArray, tau::AbstractArray) = alpha!(similar(q), t, q, tau)
function alpha!(I::AbstractArray, t::AbstractArray, q::AbstractArray, tau::AbstractArray)
    @avx @. I = t * (q / tau) * exp(-(t - tau) / tau)
    @. I = ifelse((t >= 0) && (t < Inf), I, zero(eltype(I)))

    return I
end
function alpha!(I::CuArray, t::CuArray, q::CuArray, tau::CuArray)
    @. I = (t >= 0) * (t < Inf) * t * (q / tau) * exp(-(t - tau) / tau)

    return I
end

"""
    Alpha(q, τ, dims = (1,))
    Alpha(; q = one(Float32), τ = one(Float32), dims = (1,))

Synapse whose output current is given by

``(t - t_{\\text{offset}}) * \\frac{q}{\\tau} * \\exp\\left(-\\frac{t - t_{\\text{offset}} - \\tau}{\\tau}\\right)
    \\Theta(t - t_{\\text{offset}})``

(where ``\\Theta`` is the Heaviside function).
"""
struct Alpha{T<:AbstractArray{<:Real}} <: AbstractSynapse
    offset::T
    q::T
    τ::T
end

Alpha(q, τ, dims = (1,)) = Alpha(fill(-inf(eltype(q)), dims), _fillmemaybe(q, dims), _fillmemaybe(τ, dims))
Alpha(; q = one(Float32), τ = one(Float32), dims = (1,)) = Alpha(q, τ, dims)

Base.getindex(synapse::Alpha, I...) = Alpha(synapse.offset[I...], synapse.q[I...], synapse.τ[I...])
function Base.setindex!(synapse::Alpha, v::Alpha, I...)
    synapse.offset[I...] .= v.offset[I...]
    synapse.q[I...] .= v.q[I...]
    synapse.τ[I...] .= v.τ[I...]

    return synapse
end
Base.size(synapse::Alpha) = size(synapse.offset)

"""
    excite!(synapse::Alpha, spike::Integer)

Excite `synapse` with a `spike` (`spike` == time step of spike).
"""
function excite!(synapse::Alpha, spikes)
    spiked = spikes .> 0
    synapse.offset[spiked] .= spikes[spiked]

    return synapse
end

# isactive(synapse::Alpha, t::Real; dt::Real = 1.0) = dt * (t - synapse.lastspike) <= 10 * synapse.τ
# isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Alpha} =
#     any(dt .* (t .- synapses.lastspike) .<= 10 .* synapses.τ)

"""
    evaluate!(synapse::Alpha, t; dt = 1)
    evaluate!(current, synapses::Alpha, t; dt = 1)

Evaluate an alpha synapse. See [`Alpha`](@ref).
"""
evaluate!(synapse::Alpha, t; dt = 1) = alpha((t .- synapse.offset) .* dt, synapse.q, synapse.τ)
evaluate!(current, synapse::Alpha, t; dt = 1) = alpha!(current, (t .- synapse.offset) .* dt, synapse.q, synapse.τ)

"""
    reset!(synapse::Alpha)

Reset `synapse` by setting the last pre-synaptic spike time to `-Inf`.
"""
function reset!(synapse::Alpha, mask = trues(size(synapse)))
    synapse.offset[mask] .= -Inf

    return synapse
end
